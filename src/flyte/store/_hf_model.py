"""
HuggingFace model store utilities for Flyte.

This module provides functionality to store HuggingFace models to remote storage,
with support for optional sharding using vLLM.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, get_args

if TYPE_CHECKING:
    from flyte._resources import Accelerators
    from flyte.io import Dir
    from flyte.remote import Run


# Keys for model partition metadata
ARCHITECTURE_KEY = "architecture"
TASK_KEY = "task"
FORMAT_KEY = "format"
HUGGINGFACE_SOURCE_KEY = "huggingface_source"
COMMIT_KEY = "commit"
MODALITY_KEY = "modality"
MODEL_TYPE_KEY = "model_type"
SHARD_ENGINE_KEY = "shard_engine"
SHARD_PARALLELISM_KEY = "shard_parallelism"


@dataclass
class VLLMShardArgs:
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
    max_model_len: Optional[int] = None
    file_pattern: str = "*.safetensors"
    max_file_size: int = 5 * 1024**3  # 5GB default

    def get_vllm_args(self, model_path: str) -> Dict[str, Any]:
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


@dataclass
class ShardConfig:
    """
    Configuration for model sharding.

    :param engine: The sharding engine to use (currently only "vllm" is supported).
    :param args: Arguments for the sharding engine.
    """

    engine: Literal["vllm"] = "vllm"
    args: VLLMShardArgs = field(default_factory=VLLMShardArgs)


@dataclass
class HuggingFaceModelInfo:
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
    artifact_name: Optional[str] = None
    architecture: Optional[str] = None
    task: str = "auto"
    modality: Tuple[str, ...] = ("text",)
    serial_format: Optional[str] = None
    model_type: Optional[str] = None
    short_description: Optional[str] = None
    shard_config: Optional[ShardConfig] = None


@dataclass
class StoredModelInfo:
    """
    Information about a stored model.

    :param artifact_name: Name of the stored artifact.
    :param path: Path to the stored model directory.
    :param metadata: Metadata about the stored model.
    """

    artifact_name: str
    path: str
    metadata: Dict[str, str]


# Image definitions for the store task
HF_DOWNLOAD_IMAGE_PACKAGES = [
    "huggingface-hub>=0.27.0",
    "hf-transfer>=0.1.8",
]

VLLM_SHARDING_IMAGE_PACKAGES = [
    "huggingface-hub>=0.27.0",
    "hf-transfer>=0.1.8",
    "vllm>=0.6.0",
]


def _validate_artifact_name(name: Optional[str]) -> None:
    """Validate that artifact name contains only allowed characters."""
    if name is not None and not re.match(r"^[a-zA-Z0-9_-]+$", name):
        raise ValueError(
            f"Artifact name '{name}' must only contain alphanumeric characters, underscores, and hyphens"
        )


def _lookup_huggingface_model_info(model_repo: str, commit: str, token: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Lookup HuggingFace model info from config.json.

    :param model_repo: The model repository ID.
    :param commit: The commit ID.
    :param token: HuggingFace token for private models.
    :return: Tuple of (model_type, architecture).
    """
    from huggingface_hub import hf_hub_download
    import json

    config_file = hf_hub_download(repo_id=model_repo, filename="config.json", revision=commit, token=token)
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


def _get_partition_keys(info: HuggingFaceModelInfo, commit: str) -> Dict[str, str]:
    """
    Get partition keys for a model.

    :param info: The model info.
    :param commit: The commit ID.
    :return: Dictionary of partition keys.
    """
    return {
        ARCHITECTURE_KEY: info.architecture or "",
        TASK_KEY: info.task,
        FORMAT_KEY: info.serial_format or "",
        HUGGINGFACE_SOURCE_KEY: info.repo,
        COMMIT_KEY: commit,
        MODALITY_KEY: ",".join(info.modality),
        MODEL_TYPE_KEY: info.model_type or "",
        SHARD_ENGINE_KEY: str(info.shard_config.engine) if info.shard_config else "None",
        SHARD_PARALLELISM_KEY: str(info.shard_config.args.tensor_parallel_size) if info.shard_config else "None",
    }


def _stream_to_remote_dir(
    repo_id: str,
    commit: str,
    token: Optional[str],
    remote_dir_path: str,
) -> Tuple[str, Optional[str]]:
    """
    Stream files directly from HuggingFace to a remote directory.

    :param repo_id: The HuggingFace repository ID.
    :param commit: The commit ID.
    :param token: HuggingFace token.
    :param remote_dir_path: Path to the remote directory.
    :return: Tuple of (remote_dir_path, readme_content).
    """
    from huggingface_hub import HfFileSystem
    import tempfile
    import flyte.storage as storage

    hfs = HfFileSystem(token=token)
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
        print("No README.md file found", flush=True)

    # List all files in the repo
    repo_files = hfs.ls(f"{repo_id}", revision=commit, detail=True)

    print(f"Streaming {len(repo_files)} files to {remote_dir_path}", flush=True)

    for file_info in repo_files:
        if file_info["type"] == "file":
            file_name = file_info["name"].split("/")[-1]
            remote_file_path = f"{remote_dir_path}/{file_name}"
            print(f"  Streaming {file_name}...", flush=True)

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
    token: Optional[str],
    local_dir: str,
) -> Tuple[str, Optional[str]]:
    """
    Download model snapshot to local directory.

    :param repo_id: The HuggingFace repository ID.
    :param commit: The commit ID.
    :param token: HuggingFace token.
    :param local_dir: Local directory to download to.
    :return: Tuple of (local_dir, readme_content).
    """
    from huggingface_hub import HfFileSystem, snapshot_download
    import tempfile

    card = None
    hfs = HfFileSystem(token=token)

    # Try to get README
    try:
        readme_file_details = hfs.info(f"{repo_id}/README.md", revision=commit)
        readme_name = readme_file_details["name"]
        with tempfile.NamedTemporaryFile() as temp_file:
            hfs.download(readme_name, temp_file.name, revision=commit)
            with open(temp_file.name, "r") as f:
                card = f.read()
    except FileNotFoundError:
        print("No README.md file found", flush=True)

    print(f"Downloading model from {repo_id} to {local_dir}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        revision=commit,
        local_dir=local_dir,
        token=token,
    )
    return local_dir, card


def _shard_model(
    shard_config: ShardConfig,
    model_path: str,
    output_dir: str,
) -> str:
    """
    Shard a model using vLLM.

    :param shard_config: Sharding configuration.
    :param model_path: Path to the model to shard.
    :param output_dir: Directory to save sharded model.
    :return: Path to sharded model directory.
    """
    import shutil

    from vllm import LLM

    assert shard_config.engine == "vllm", "'vllm' is the only supported sharding engine for now"

    # Create LLM instance
    llm = LLM(**shard_config.args.get_vllm_args(model_path))
    print(f"LLM initialized: {llm}")

    # Check which engine version is being used
    is_v1_engine = hasattr(llm.llm_engine, "engine_core")

    if is_v1_engine:
        # For V1 engine
        print("Using V1 engine save path")
        llm.llm_engine.engine_core.save_sharded_state(
            path=output_dir,
            pattern=shard_config.args.file_pattern,
            max_size=shard_config.args.max_file_size,
        )
    else:
        # For V0 engine
        print("Using V0 engine save path")
        model_executor = llm.llm_engine.model_executor
        model_executor.save_sharded_state(
            path=output_dir,
            pattern=shard_config.args.file_pattern,
            max_size=shard_config.args.max_file_size,
        )

    # Copy metadata files to output directory
    print(f"Copying metadata files to {output_dir}")
    for file in os.listdir(model_path):
        if os.path.splitext(file)[1] not in (".bin", ".pt", ".safetensors"):
            src_path = os.path.join(model_path, file)
            dst_path = os.path.join(output_dir, file)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy(src_path, dst_path)

    return output_dir


def _get_latest_commit(repo_id: str, token: Optional[str]) -> str:
    """
    Get the latest commit ID for a HuggingFace repository.

    :param repo_id: The HuggingFace repository ID.
    :param token: HuggingFace token.
    :return: The latest commit ID.
    """
    from huggingface_hub import list_repo_commits, repo_exists

    if not repo_exists(repo_id, token=token):
        raise ValueError(f"Repository {repo_id} does not exist in HuggingFace.")

    commit = list_repo_commits(repo_id, token=token)[0]
    return commit.commit_id


def hf_model(
    repo: str,
    *,
    artifact_name: Optional[str] = None,
    architecture: Optional[str] = None,
    task: str = "auto",
    modality: Tuple[str, ...] = ("text",),
    serial_format: Optional[str] = None,
    model_type: Optional[str] = None,
    short_description: Optional[str] = None,
    shard_config: Optional[ShardConfig] = None,
    hf_token_key: str = "HF_TOKEN",
    cpu: Optional[str] = None,
    mem: Optional[str] = None,
    ephemeral_storage: Optional[str] = None,
    accelerator: Optional["Accelerators"] = None,
    project: Optional[str] = None,
    domain: Optional[str] = None,
    wait: bool = False,
    force: int = 0,
) -> Run:
    """
    Store a HuggingFace model to remote storage.

    This function downloads a model from the HuggingFace Hub and stores it in
    remote storage. It supports optional sharding using vLLM for large models.

    The store behavior follows this priority:
    1. If the model isn't being sharded, stream files directly to remote storage.
    2. If streaming fails, fall back to downloading a snapshot and uploading.
    3. If sharding is configured, download locally, shard with vLLM, then upload.

    Example usage:

    ```python
    import flyte

    flyte.init(endpoint="my-flyte-endpoint")

    # Store a model without sharding
    run = flyte.store.hf_model(
        repo="meta-llama/Llama-2-7b-hf",
        hf_token_key="HF_TOKEN",
    )
    run.wait()

    # Store and shard a model
    from flyte.store import ShardConfig, VLLMShardArgs

    run = flyte.store.hf_model(
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
    :param cpu: CPU request for the store task (e.g., '2').
    :param mem: Memory request for the store task (e.g., '16Gi').
    :param ephemeral_storage: Ephemeral storage request (e.g., '100Gi').
    :param accelerator: Accelerator type in format '{type}:{quantity}' (e.g., 'A100:8', 'L4:1').
    :param project: Project to run the store task in.
    :param domain: Domain to run the store task in.
    :param wait: Whether to wait for the store task to complete. Default: False.
    :param force: Force re-store. Increment to force a new store. Default: 0.

    :return: A Run object representing the store task execution.
    """
    import flyte
    from flyte import Resources, Secret, TaskEnvironment
    from flyte._initialize import get_init_config
    from flyte._resources import Accelerators

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

    # Validate accelerator if provided
    if accelerator is not None and accelerator not in get_args(Accelerators):
        raise ValueError(
            f"Invalid accelerator: {accelerator}. Must be one of the valid Accelerators types "
            f"in format '{{type}}:{{quantity}}' (e.g., 'A100:8', 'L4:1')"
        )

    # Build resources - use accelerator directly since Resources.gpu accepts Accelerators strings
    resources = Resources(
        cpu=cpu or "2",
        memory=mem or "8Gi",
        gpu=accelerator,  # Accelerators string like "A100:8" or "L4:1"
        disk=ephemeral_storage or "50Gi",
    )

    # Select image based on whether sharding is needed
    if shard_config is not None:
        image = flyte.Image.from_debian_base().with_packages(VLLM_SHARDING_IMAGE_PACKAGES)
    else:
        image = flyte.Image.from_debian_base().with_packages(HF_DOWNLOAD_IMAGE_PACKAGES)

    # Build environment kwargs
    env_kwargs: Dict[str, Any] = {
        "name": "hf-model-store",
        "image": image,
        "resources": resources,
        "secrets": [Secret(key=hf_token_key)],
    }

    env = TaskEnvironment(**env_kwargs)

    @env.task(cache="auto")
    def store_hf_model_task(
        info: HuggingFaceModelInfo,
        hf_token_key: str,
        force: int,
    ) -> Dir:
        """Task to store a HuggingFace model."""
        # All imports inside the task body
        import os
        import tempfile

        from huggingface_hub import HfFileSystem, list_repo_commits, repo_exists, snapshot_download
        import flyte
        from flyte import ctx
        from flyte.io import Dir
        import flyte.storage as storage

        # Get HF token from secrets
        token = ctx.secrets.get(key=hf_token_key)

        # Validate repo exists and get latest commit
        if not repo_exists(info.repo, token=token):
            raise ValueError(f"Repository {info.repo} does not exist in HuggingFace.")

        commit = list_repo_commits(info.repo, token=token)[0].commit_id
        print(f"Latest commit: {commit}", flush=True)

        # Lookup model info if not provided
        if not info.model_type or not info.architecture:
            print("Looking up HuggingFace model info...", flush=True)
            try:
                import json
                from huggingface_hub import hf_hub_download

                config_file = hf_hub_download(
                    repo_id=info.repo,
                    filename="config.json",
                    revision=commit,
                    token=token,
                )
                with open(config_file, "r") as f:
                    j = json.load(f)
                    if not info.architecture:
                        arch = j.get("architecture", None) or j.get("architectures", None)
                        if isinstance(arch, list):
                            arch = ",".join(arch)
                        info.architecture = arch
                    if not info.model_type:
                        info.model_type = j.get("model_type", "custom")
            except Exception as e:
                print(f"Warning: Could not lookup model info: {e}", flush=True)
                info.model_type = info.model_type or "custom"
                info.architecture = info.architecture or "custom"

        print(f"Model type: {info.model_type}, architecture: {info.architecture}", flush=True)

        # Determine artifact name
        if info.artifact_name is None:
            artifact_name = info.repo.split("/")[-1].replace(".", "-")
        else:
            artifact_name = info.artifact_name

        card = None

        # If sharding is needed, we must download locally first
        if info.shard_config is not None:
            print(f"Sharding requested with {info.shard_config.engine} engine", flush=True)

            # Download to local temp directory
            with tempfile.TemporaryDirectory() as local_model_dir:
                # Download snapshot
                hfs = HfFileSystem(token=token)
                try:
                    readme_info = hfs.info(f"{info.repo}/README.md", revision=commit)
                    with tempfile.NamedTemporaryFile() as temp_file:
                        hfs.download(readme_info["name"], temp_file.name, revision=commit)
                        with open(temp_file.name, "r") as f:
                            card = f.read()
                except FileNotFoundError:
                    print("No README.md found", flush=True)

                print(f"Downloading model to {local_model_dir}", flush=True)
                snapshot_download(
                    repo_id=info.repo,
                    revision=commit,
                    local_dir=local_model_dir,
                    token=token,
                )

                # Shard the model
                import shutil
                from vllm import LLM

                sharded_dir = tempfile.mkdtemp()
                print(f"Sharding model to {sharded_dir}", flush=True)

                llm = LLM(**info.shard_config.args.get_vllm_args(local_model_dir))

                is_v1_engine = hasattr(llm.llm_engine, "engine_core")
                if is_v1_engine:
                    llm.llm_engine.engine_core.save_sharded_state(
                        path=sharded_dir,
                        pattern=info.shard_config.args.file_pattern,
                        max_size=info.shard_config.args.max_file_size,
                    )
                else:
                    model_executor = llm.llm_engine.model_executor
                    model_executor.save_sharded_state(
                        path=sharded_dir,
                        pattern=info.shard_config.args.file_pattern,
                        max_size=info.shard_config.args.max_file_size,
                    )

                # Copy metadata files
                for file in os.listdir(local_model_dir):
                    if os.path.splitext(file)[1] not in (".bin", ".pt", ".safetensors"):
                        src = os.path.join(local_model_dir, file)
                        dst = os.path.join(sharded_dir, file)
                        if os.path.isdir(src):
                            shutil.copytree(src, dst, dirs_exist_ok=True)
                        else:
                            shutil.copy(src, dst)

                # Upload sharded model
                print(f"Uploading sharded model...", flush=True)
                result_dir = Dir.from_local_sync(sharded_dir)
                shutil.rmtree(sharded_dir)

        else:
            # Try direct streaming first
            try:
                print("Attempting direct streaming to remote storage...", flush=True)
                remote_path = ctx.raw_data.get_random_remote_path(artifact_name)
                fs = storage.get_underlying_filesystem(path=remote_path)
                fs.makedirs(remote_path, exist_ok=True)

                hfs = HfFileSystem(token=token)

                # Get README if available
                try:
                    readme_info = hfs.info(f"{info.repo}/README.md", revision=commit)
                    with tempfile.NamedTemporaryFile() as temp_file:
                        hfs.download(readme_info["name"], temp_file.name, revision=commit)
                        with open(temp_file.name, "r") as f:
                            card = f.read()
                except FileNotFoundError:
                    print("No README.md found", flush=True)

                # List and stream all files
                repo_files = hfs.ls(info.repo, revision=commit, detail=True)
                for file_info in repo_files:
                    if file_info["type"] == "file":
                        file_name = file_info["name"].split("/")[-1]
                        remote_file_path = f"{remote_path}/{file_name}"
                        print(f"  Streaming {file_name}...", flush=True)

                        with hfs.open(file_info["name"], "rb", revision=commit) as src:
                            with fs.open(remote_file_path, "wb") as dst:
                                chunk_size = 64 * 1024 * 1024  # 64MB chunks
                                while True:
                                    chunk = src.read(chunk_size)
                                    if not chunk:
                                        break
                                    dst.write(chunk)

                result_dir = Dir.from_existing_remote(remote_path)
                print(f"Direct streaming completed to {remote_path}", flush=True)

            except Exception as e:
                print(f"Direct streaming failed: {e}", flush=True)
                print("Falling back to snapshot download...", flush=True)

                # Fallback: download snapshot and upload
                with tempfile.TemporaryDirectory() as local_model_dir:
                    hfs = HfFileSystem(token=token)
                    try:
                        readme_info = hfs.info(f"{info.repo}/README.md", revision=commit)
                        with tempfile.NamedTemporaryFile() as temp_file:
                            hfs.download(readme_info["name"], temp_file.name, revision=commit)
                            with open(temp_file.name, "r") as f:
                                card = f.read()
                    except FileNotFoundError:
                        print("No README.md found", flush=True)

                    print(f"Downloading snapshot to {local_model_dir}", flush=True)
                    snapshot_download(
                        repo_id=info.repo,
                        revision=commit,
                        local_dir=local_model_dir,
                        token=token,
                    )

                    print("Uploading to remote storage...", flush=True)
                    result_dir = Dir.from_local_sync(local_model_dir)

        print(f"Model stored successfully at {result_dir.path}", flush=True)
        return result_dir

    # Get config for project/domain
    cfg = get_init_config()
    run_project = project or cfg.project
    run_domain = domain or cfg.domain

    # Run the task
    run = flyte.with_runcontext(
        project=run_project,
        domain=run_domain,
    ).run(store_hf_model_task, info, hf_token_key, force)

    if wait:
        run.wait()

    return run
