"""
HuggingFace model prefetch utilities for Flyte.

This module provides functionality to prefetch HuggingFace models to remote storage,
with support for optional sharding using vLLM.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import typing
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from flyte._logging import logger
from flyte._resources import Resources
from flyte._task_environment import TaskEnvironment
from flyte.io import Dir

if TYPE_CHECKING:
    from flyte.remote import Run


DEFAULT_SHARD_PATTERN = "model-rank-{rank}-part-{part}.safetensors"


class VLLMShardArgs(BaseModel):
    """
    Arguments for sharding a model using vLLM.

    :param tensor_parallel_size: Number of tensor parallel workers.
    :param dtype: Data type for model weights.
    :param trust_remote_code: Whether to trust remote code from HuggingFace.
    :param max_model_len: Maximum model context length.
    :param file_pattern: Pattern for sharded weight files.
    :param max_file_size: Maximum size for each sharded file.
    """

    tensor_parallel_size: int = 1
    dtype: str = "auto"
    trust_remote_code: bool = True
    max_model_len: int | None = None
    file_pattern: str | None = DEFAULT_SHARD_PATTERN
    max_file_size: int = 5 * 1024**3  # 5GB default

    def get_vllm_args(self, model_path: str) -> dict[str, Any]:
        """Get arguments dict for vLLM LLM constructor."""
        args = {
            "model": model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.max_model_len is not None:
            args["max_model_len"] = self.max_model_len
        return args


class ShardConfig(BaseModel):
    """
    Configuration for model sharding.

    :param engine: The sharding engine to use (currently only "vllm" is supported).
    :param args: Arguments for the sharding engine.
    """

    engine: Literal["vllm"] = "vllm"
    args: VLLMShardArgs = Field(default_factory=VLLMShardArgs)


class HuggingFaceModelInfo(BaseModel):
    """
    Information about a HuggingFace model to store.

    :param repo: The HuggingFace repository ID (e.g., 'meta-llama/Llama-2-7b-hf').
    :param artifact_name: Optional name for the stored artifact. If not provided,
        the repo name will be used (with '.' replaced by '-').
    :param architecture: Model architecture from HuggingFace config.json.
    :param task: Model task (e.g., 'generate', 'classify', 'embed').
    :param modality: Modalities supported by the model (e.g., 'text', 'image').
    :param serial_format: Model serialization format (e.g., 'safetensors', 'onnx').
    :param model_type: Model type (e.g., 'transformer', 'custom').
    :param short_description: Short description of the model.
    :param shard_config: Optional configuration for model sharding.
    """

    repo: str
    artifact_name: str | None = None
    architecture: str | None = None
    task: str = "auto"
    modality: tuple[str, ...] = ("text",)
    serial_format: str | None = None
    model_type: str | None = None
    short_description: str | None = None
    shard_config: ShardConfig | None = None


class StoredModelInfo(BaseModel):
    """
    Information about a stored model.

    :param artifact_name: Name of the stored artifact.
    :param path: Path to the stored model directory.
    :param metadata: Metadata about the stored model.
    """

    artifact_name: str
    path: str
    metadata: dict[str, str]


# Image definitions for the store task
HF_DOWNLOAD_IMAGE_PACKAGES = [
    "huggingface-hub>=0.27.0",
    "hf-transfer>=0.1.8",
    "markdown>=3.10",
]

VLLM_SHARDING_IMAGE_PACKAGES = [
    *HF_DOWNLOAD_IMAGE_PACKAGES,
    "vllm>=0.11.0",
]


def _validate_artifact_name(name: str | None) -> None:
    """Validate that artifact name contains only allowed characters."""
    if name is not None and not re.match(r"^[a-zA-Z0-9_-]+$", name):
        raise ValueError(f"Artifact name '{name}' must only contain alphanumeric characters, underscores, and hyphens")


def _lookup_huggingface_model_info(model_repo: str, commit: str, token: str | None) -> tuple[str | None, str | None]:
    """
    Lookup HuggingFace model info from config.json.

    :param model_repo: The model repository ID.
    :param commit: The commit ID.
    :param token: HuggingFace token for private models.
    :return: Tuple of (model_type, architecture).
    """
    import json

    import huggingface_hub

    config_file = huggingface_hub.hf_hub_download(
        repo_id=model_repo, filename="config.json", revision=commit, token=token
    )
    arch = None
    model_type = None
    with open(config_file, "r") as f:
        j = json.load(f)
        arch = j.get("architecture", None)
        if arch is None:
            arch = j.get("architectures", None)
            if arch:
                arch = ",".join(arch)
        model_type = j.get("model_type", None)
    return model_type, arch


def _stream_to_remote_dir(
    repo_id: str,
    commit: str,
    token: str | None,
    remote_dir_path: str,
) -> tuple[str, str | None]:
    """
    Stream files directly from HuggingFace to a remote directory.

    :param repo_id: The HuggingFace repository ID.
    :param commit: The commit ID.
    :param token: HuggingFace token.
    :param remote_dir_path: Path to the remote directory.
    :return: Tuple of (remote_dir_path, readme_content).
    """
    import huggingface_hub

    import flyte.storage as storage

    hfs = huggingface_hub.HfFileSystem(token=token)
    fs = storage.get_underlying_filesystem(path=remote_dir_path)
    card = None

    # Try to get README
    try:
        readme_file_details = hfs.info(f"{repo_id}/README.md", revision=commit)
        readme_name = readme_file_details["name"]
        with tempfile.NamedTemporaryFile() as temp_file:
            hfs.download(readme_name, temp_file.name, revision=commit)
            with open(temp_file.name, "r") as f:
                card = f.read()
    except FileNotFoundError:
        logger.info("No README.md file found")

    # List all files in the repo
    repo_files = hfs.ls(f"{repo_id}", revision=commit, detail=True)

    logger.info(f"Streaming {len(repo_files)} files to {remote_dir_path}")

    for file_info in repo_files:
        if isinstance(file_info, str):
            logger.info(f"  Skipping {file_info}...")
            continue
        if file_info["type"] == "file":
            file_name = file_info["name"].split("/")[-1]
            remote_file_path = f"{remote_dir_path}/{file_name}"
            logger.info(f"  Streaming {file_name}...")

            # Stream file content directly to remote
            with hfs.open(file_info["name"], "rb", revision=commit) as src:
                with fs.open(remote_file_path, "wb") as dst:
                    # Stream in chunks
                    chunk_size = 64 * 1024 * 1024  # 64MB chunks
                    while True:
                        chunk = src.read(chunk_size)
                        if not chunk:
                            break
                        dst.write(chunk)

    return remote_dir_path, card


def _download_snapshot_to_local(
    repo_id: str,
    commit: str,
    token: str | None,
    local_dir: str,
) -> tuple[str, str | None]:
    """
    Download model snapshot to local directory.

    :param repo_id: The HuggingFace repository ID.
    :param commit: The commit ID.
    :param token: HuggingFace token.
    :param local_dir: Local directory to download to.
    :return: Tuple of (local_dir, readme_content).
    """
    import huggingface_hub

    card = None
    hfs = huggingface_hub.HfFileSystem(token=token)

    # Try to get README
    try:
        readme_file_details = hfs.info(f"{repo_id}/README.md", revision=commit)
        readme_name = readme_file_details["name"]
        with tempfile.NamedTemporaryFile() as temp_file:
            hfs.download(readme_name, temp_file.name, revision=commit)
            with open(temp_file.name, "r") as f:
                card = f.read()
    except FileNotFoundError:
        logger.info("No README.md file found")

    logger.info(f"Downloading model from {repo_id} to {local_dir}")
    huggingface_hub.snapshot_download(
        repo_id=repo_id,
        revision=commit,
        local_dir=local_dir,
        token=token,
    )
    return local_dir, card


def _shard_model(
    repo: str,
    commit: str,
    shard_config: ShardConfig,
    token: str,
    model_path: str,
    output_dir: str,
) -> tuple[str, str | None]:
    """
    Shard a model using vLLM.

    :param shard_config: Sharding configuration.
    :param model_path: Path to the model to shard.
    :param output_dir: Directory to save sharded model.
    :return: Path to sharded model directory.
    """
    import huggingface_hub
    import vllm

    assert shard_config.engine == "vllm", "'vllm' is the only supported sharding engine for now"

    # Download snapshot
    hfs = huggingface_hub.HfFileSystem(token=token)
    try:
        readme_info = hfs.info(f"{repo}/README.md", revision=commit)
        with tempfile.NamedTemporaryFile() as temp_file:
            hfs.download(readme_info["name"], temp_file.name, revision=commit)
            with open(temp_file.name, "r") as f:
                card = f.read()
    except FileNotFoundError:
        logger.warning("No README.md found")

    logger.info(f"Downloading model to {model_path}")
    huggingface_hub.snapshot_download(
        repo_id=repo,
        revision=commit,
        local_dir=model_path,
        token=token,
    )

    # Create LLM instance
    llm = vllm.LLM(**shard_config.args.get_vllm_args(model_path))
    logger.info(f"LLM initialized: {llm}")

    llm.llm_engine.engine_core.save_sharded_state(
        path=output_dir,
        pattern=shard_config.args.file_pattern,
        max_size=shard_config.args.max_file_size,
    )

    # Copy metadata files to output directory
    logger.info(f"Copying metadata files to {output_dir}")
    for file in os.listdir(model_path):
        if os.path.splitext(file)[1] not in (".bin", ".pt", ".safetensors"):
            src_path = os.path.join(model_path, file)
            dst_path = os.path.join(output_dir, file)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy(src_path, dst_path)

    return output_dir, card


# NOTE: the info argument is a json string instead of a HuggingFaceModelInfo
# object because the type engine cannot handle nested pydantic or dataclass
# objects when run in interactive mode.
def store_hf_model_task(info: str, raw_data_path: str | None = None) -> Dir:
    """Task to store a HuggingFace model."""

    import huggingface_hub

    import flyte.report

    # Get HF token from secrets
    token = os.environ.get("HF_TOKEN")
    assert token is not None, "HF_TOKEN environment variable is not set"

    # Validate repo exists and get latest commit
    _info: HuggingFaceModelInfo = HuggingFaceModelInfo.model_validate_json(info)
    if not huggingface_hub.repo_exists(_info.repo, token=token):
        raise ValueError(f"Repository {_info.repo} does not exist in HuggingFace.")

    commit = huggingface_hub.list_repo_commits(_info.repo, token=token)[0].commit_id
    logger.info(f"Latest commit: {commit}")

    # Lookup model info if not provided
    if not _info.model_type or not _info.architecture:
        logger.info("Looking up HuggingFace model info...")
        try:
            _info.model_type, _info.architecture = _lookup_huggingface_model_info(_info.repo, commit, token)
        except Exception as e:
            logger.warning(f"Warning: Could not lookup model info: {e}")
            _info.model_type = "custom"
            _info.architecture = "custom"

    logger.info(f"Model type: {_info.model_type}, architecture: {_info.architecture}")

    # Determine artifact name
    if _info.artifact_name is None:
        artifact_name = _info.repo.split("/")[-1].replace(".", "-")
    else:
        artifact_name = _info.artifact_name

    card = None
    result_dir: Dir

    # If sharding is needed, we must download locally first
    if _info.shard_config is not None:
        logger.info(f"Sharding requested with {_info.shard_config.engine} engine")

        # Download to local temp directory
        sharded_dir = tempfile.mkdtemp()
        with tempfile.TemporaryDirectory() as local_model_dir:
            sharded_dir, card = _shard_model(
                _info.repo, commit, _info.shard_config, token, local_model_dir, sharded_dir
            )

            # Upload sharded model
            logger.info("Uploading sharded model...")
            result_dir = Dir.from_local_sync(sharded_dir, remote_destination=raw_data_path)

    else:
        # Try direct streaming first
        try:
            logger.info("Attempting direct streaming to remote storage...")

            if raw_data_path is not None:
                remote_path = raw_data_path
            else:
                remote_path = flyte.ctx().raw_data_path.get_random_remote_path(artifact_name)  # type: ignore [union-attr]

            remote_path, card = _stream_to_remote_dir(_info.repo, commit, token, remote_path)
            result_dir = Dir.from_existing_remote(remote_path)
            logger.info(f"Direct streaming completed to {remote_path}")

        except Exception as e:
            logger.error(f"Direct streaming failed: {e}")
            logger.error("Falling back to snapshot download...")

            # Fallback: download snapshot and upload
            with tempfile.TemporaryDirectory() as local_model_dir:
                _local_model_dir, card = _download_snapshot_to_local(_info.repo, commit, token, local_model_dir)
                result_dir = Dir.from_local_sync(_local_model_dir, remote_destination=raw_data_path)

    # create report from the markdown `card`
    if card:
        # Try to convert markdown to HTML for richer presentation, fallback to plain text
        try:
            # Try to import markdown if available (don't add import; just use if exists)
            import markdown

            report = markdown.markdown(card)
        except Exception:
            report = card  # fallback to plain markdown content
        flyte.report.log(report)
        flyte.report.flush()

    logger.info(f"Model stored successfully at {result_dir.path}")
    return result_dir


def hf_model(
    repo: str,
    *,
    raw_data_path: str | None = None,
    artifact_name: str | None = None,
    architecture: str | None = None,
    task: str = "auto",
    modality: tuple[str, ...] = ("text",),
    serial_format: str | None = None,
    model_type: str | None = None,
    short_description: str | None = None,
    shard_config: ShardConfig | None = None,
    hf_token_key: str = "HF_TOKEN",
    resources: Resources = Resources(cpu="2", memory="8Gi", disk="50Gi"),
    force: int = 0,
) -> Run:
    """
    Store a HuggingFace model to remote storage.

    This function downloads a model from the HuggingFace Hub and prefetches it to
    remote storage. It supports optional sharding using vLLM for large models.

    The prefetch behavior follows this priority:
    1. If the model isn't being sharded, stream files directly to remote storage.
    2. If streaming fails, fall back to downloading a snapshot and uploading.
    3. If sharding is configured, download locally, shard with vLLM, then upload.

    Example usage:

    ```python
    import flyte

    flyte.init(endpoint="my-flyte-endpoint")

    # Store a model without sharding
    run = flyte.prefetch.hf_model(
        repo="meta-llama/Llama-2-7b-hf",
        hf_token_key="HF_TOKEN",
    )
    run.wait()

    # Prefetch and shard a model
    from flyte.prefetch import ShardConfig, VLLMShardArgs

    run = flyte.prefetch.hf_model(
        repo="meta-llama/Llama-2-70b-hf",
        shard_config=ShardConfig(
            engine="vllm",
            args=VLLMShardArgs(tensor_parallel_size=8),
        ),
        accelerator="A100:8",
        hf_token_key="HF_TOKEN",
    )
    run.wait()
    ```

    :param repo: The HuggingFace repository ID (e.g., 'meta-llama/Llama-2-7b-hf').
    :param artifact_name: Optional name for the stored artifact. If not provided,
        the repo name will be used (with '.' replaced by '-').
    :param architecture: Model architecture from HuggingFace config.json.
    :param task: Model task (e.g., 'generate', 'classify', 'embed'). Default: 'auto'.
    :param modality: Modalities supported by the model. Default: ('text',).
    :param serial_format: Model serialization format (e.g., 'safetensors', 'onnx').
    :param model_type: Model type (e.g., 'transformer', 'custom').
    :param short_description: Short description of the model.
    :param shard_config: Optional configuration for model sharding with vLLM.
    :param hf_token_key: Name of the secret containing the HuggingFace token. Default: 'HF_TOKEN'.
    :param cpu: CPU request for the prefetch task (e.g., '2').
    :param mem: Memory request for the prefetch task (e.g., '16Gi').
    :param disk: Disk storage request (e.g., '100Gi').
    :param gpu: Accelerator type in format '{type}:{quantity}' (e.g., 'A100:8', 'L4:1').
    :param shm: Shared memory request (e.g., '100Gi', 'auto').
    :param wait: Whether to wait for the prefetch task to complete. Default: False.
    :param force: Force re-prefetch. Increment to force a new prefetch. Default: 0.

    :return: A Run object representing the prefetch task execution.
    """
    import flyte
    from flyte import Secret
    from flyte.remote import Run

    _validate_artifact_name(artifact_name)

    info = HuggingFaceModelInfo(
        repo=repo,
        artifact_name=artifact_name,
        architecture=architecture,
        task=task,
        modality=modality,
        serial_format=serial_format,
        model_type=model_type,
        short_description=short_description,
        shard_config=shard_config,
    )

    # Select image based on whether sharding is needed
    if shard_config is not None:
        image = (
            flyte.Image.from_debian_base(name="prefetch-hf-model-image")
            .with_apt_packages("gcc", "wget")
            .with_commands(
                [
                    "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
                    "dpkg -i cuda-keyring_1.1-1_all.deb",
                    "apt-get update",
                    "apt-get install -y cuda-toolkit-12-9",
                ]
            )
            .with_env_vars(
                {
                    "CUDA_HOME": "/usr/local/cuda-12.9",
                    "LD_LIBRARY_PATH": "/usr/local/cuda-12.9/lib64/stubs",
                    "VLLM_USE_V1": "1",
                }
            )
            .with_pip_packages(*VLLM_SHARDING_IMAGE_PACKAGES)
        )
    else:
        image = flyte.Image.from_debian_base(name="prefetch-hf-model-image").with_pip_packages(
            *HF_DOWNLOAD_IMAGE_PACKAGES
        )

    # Create a task from the module-level function with the configured environment
    disable_run_cache = force > 0
    env = TaskEnvironment(
        name="prefetch-hf-model",
        image=image,
        resources=resources,
        secrets=[Secret(key=hf_token_key, as_env_var="HF_TOKEN")],
    )
    task = env.task(report=True)(store_hf_model_task)  # type: ignore [assignment]
    run = flyte.with_runcontext(interactive_mode=True, disable_run_cache=disable_run_cache).run(
        task, info.model_dump_json(), raw_data_path
    )
    return typing.cast(Run, run)
