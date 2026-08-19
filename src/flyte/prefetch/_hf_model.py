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

# Minimal readable wrapper for HTML model cards rendered from the repo README.
_CARD_HTML_PREFIX = (
    "<!DOCTYPE html><html><head><meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width, initial-scale=1'><style>"
    "body{font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;"
    "font-size:15px;line-height:1.65;max-width:860px;margin:0 auto;padding:40px 32px;"
    "color:#24292f;background:#fff}"
    "a{color:#7652a2}"
    "h1,h2{border-bottom:1px solid #eaecef;padding-bottom:.3em;letter-spacing:-0.3px}"
    "h1,h2,h3{margin-top:1.4em;margin-bottom:.6em}"
    "code{background:#f4f2fa;border-radius:6px;padding:2px 6px;font-size:85%;"
    "font-family:ui-monospace,'SF Mono',Menlo,Consolas,monospace}"
    "pre{background:#f6f8fa;border-radius:8px;padding:16px;overflow-x:auto;line-height:1.45}"
    "pre code{background:transparent;padding:0;font-size:13px}"
    "blockquote{border-left:4px solid #d8dee4;margin-left:0;padding-left:16px;color:#57606a}"
    "img{max-width:100%}ul,ol{padding-left:1.6em}li{margin:.2em 0}"
    "table{border-collapse:collapse;display:block;overflow-x:auto}"
    "td,th{border:1px solid #d8dee4;padding:6px 12px}th{background:#f6f8fa}"
    "hr{border:none;border-top:1px solid #eaecef;margin:24px 0}"
    "</style></head><body>"
)
_CARD_HTML_SUFFIX = "</body></html>"


class VLLMShardArgs(BaseModel):
    """
    Arguments for sharding a model using vLLM.

    Args:
        tensor_parallel_size: Number of tensor parallel workers.
        dtype: Data type for model weights.
        trust_remote_code: Whether to trust remote code from HuggingFace.
        max_model_len: Maximum model context length.
        file_pattern: Pattern for sharded weight files.
        max_file_size: Maximum size for each sharded file.
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

    Args:
        engine: The sharding engine to use (currently only "vllm" is supported).
        args: Arguments for the sharding engine.
    """

    engine: Literal["vllm"] = "vllm"
    args: VLLMShardArgs = Field(default_factory=VLLMShardArgs)


class HuggingFaceModelInfo(BaseModel):
    """
    Information about a HuggingFace model to store.

    Args:
        repo: The HuggingFace repository ID (e.g., 'meta-llama/Llama-2-7b-hf').
        artifact_name: Optional name for the stored artifact. If not provided,
            the repo name will be used (with '.' replaced by '-').
        architecture: Model architecture from HuggingFace config.json.
        task: Model task (e.g., 'generate', 'classify', 'embed').
        modality: Modalities supported by the model (e.g., 'text', 'image').
        serial_format: Model serialization format (e.g., 'safetensors', 'onnx').
        model_type: Model type (e.g., 'transformer', 'custom').
        short_description: Short description of the model.
        shard_config: Optional configuration for model sharding.
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

    Args:
        artifact_name: Name of the stored artifact.
        path: Path to the stored model directory.
        metadata: Metadata about the stored model.
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

#: Serving facts, as versioned JSON, under the same reserved `flyte.io/`
#: namespace as `KIND_KEY` -- see the rationale there. Prefetch is the only
#: producer: these are measurements of one specific checkpoint, and code that
#: has not looked at the weights has no business claiming them.
#:
#: One nested blob rather than ~15 flat attrs, because `attrs` is
#: map<string,string> and spreading the schema across the key set would freeze
#: it. Consumers refuse a version they do not recognise rather than reading it
#: partially: a half-understood model sizes wrong, and sizing wrong is worse
#: than declining to size at all.
SERVING_ATTR_KEY = "flyte.io/serving"
SERVING_FACTS_VERSION = 1

#: Legacy config spellings for the geometry fields, tried in order after the
#: modern name. GPT-2-family configs still ship `n_layer`/`n_head`/`n_embd`, and
#: transformers only reconciles them through each config class's `attribute_map`
#: -- reading the raw JSON, as we do, sees the original names.
#:
#: Worth the table: a missed layer or head count silently zeroes the KV-cache
#: term, and that under-estimates VRAM, which is the direction that OOMs a
#: deploy rather than merely wasting a GPU.
_CONFIG_ALIASES = {
    "num_hidden_layers": ("n_layer", "num_layers", "n_layers"),
    "num_attention_heads": ("n_head", "n_heads", "encoder_attention_heads"),
    "hidden_size": ("n_embd", "d_model", "hidden_dim"),
    "max_position_embeddings": ("n_positions", "n_ctx", "max_seq_len", "seq_length"),
    "head_dim": ("d_kv", "attention_head_dim"),
    "num_key_value_heads": ("num_kv_heads", "n_kv_heads"),
}

#: Bytes per element for the dtype names the Hub's safetensors scan reports.
#: Unrecognised dtypes fall back to 2, which is what every current 16-bit
#: checkpoint uses -- conservative rather than absent.
_DTYPE_BYTES = {
    "f64": 8,
    "f32": 4,
    "f16": 2,
    "bf16": 2,
    "f8_e4m3": 1,
    "f8_e5m2": 1,
    "i64": 8,
    "i32": 4,
    "i16": 2,
    "i8": 1,
    "u8": 1,
    "bool": 1,
}


def _validate_artifact_name(name: str | None) -> None:
    """Validate that artifact name contains only allowed characters."""
    if name is not None and not re.match(r"^[a-zA-Z0-9_-]+$", name):
        raise ValueError(f"Artifact name '{name}' must only contain alphanumeric characters, underscores, and hyphens")


def _lookup_huggingface_model_info(
    model_repo: str, commit: str, token: str | None
) -> tuple[str | None, str | None, dict[str, Any]]:
    """
    Lookup HuggingFace model info from config.json.

    Args:
        model_repo: The model repository ID.
        commit: The commit ID.
        token: HuggingFace token for private models.

    Returns:
        Tuple of (model_type, architecture, raw config). The raw config is
        returned as well because the serving facts are derived almost entirely
        from it, and it costs one ~2KB download to fetch.
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
    return model_type, arch, j


def _hf_weight_stats(repo_id: str, commit: str, token: str | None) -> tuple[int, int, bool, str]:
    """
    Parameter count, weight bytes, and whether the checkpoint can be streamed.

    Read from the Hub's metadata rather than from the weights: every number here
    is available before a single byte of the model is downloaded, which is what
    lets a caller size a model it has not fetched yet.

    Returns (params_total, weight_bytes, streamable, stream_blocked_reason).
    """
    import huggingface_hub

    params_total = 0
    weight_bytes = 0

    # Prefer the Hub's own safetensors scan: it reports parameter counts per
    # dtype, which converts to bytes exactly and sidesteps the file-selection
    # traps that summing blobs runs into (repos shipping both .bin and
    # .safetensors, or an extra fp8/ subdirectory, double-count).
    try:
        info = huggingface_hub.HfApi(token=token).model_info(repo_id, revision=commit)
        scan = getattr(info, "safetensors", None)
        if scan:
            params_total = int(getattr(scan, "total", 0) or 0)
            for dtype, count in (getattr(scan, "parameters", None) or {}).items():
                weight_bytes += int(count) * _DTYPE_BYTES.get(str(dtype).lower(), 2)
    except Exception as e:
        logger.info(f"HuggingFace safetensors scan unavailable for {repo_id}: {e}")

    # The listing is needed regardless, to decide streamability -- the serving
    # loader reads safetensors and nothing else, so a repo without them cannot
    # be served no matter how well it sizes.
    safetensors_bytes = 0
    has_safetensors = False
    try:
        hfs = huggingface_hub.HfFileSystem(token=token)
        for file_info in hfs.ls(repo_id, revision=commit, detail=True):
            if isinstance(file_info, str) or file_info.get("type") != "file":
                continue
            name = str(file_info.get("name", ""))
            if name.endswith(".safetensors"):
                has_safetensors = True
                safetensors_bytes += int(file_info.get("size") or 0)
    except Exception as e:
        # Not fatal: without the listing we cannot prove the checkpoint is
        # unstreamable, and refusing to publish over a failed metadata call
        # would be a worse outcome than publishing without facts.
        logger.warning(f"Could not list files for {repo_id}: {e}")
        return params_total, weight_bytes, False, "the model's files could not be listed from HuggingFace"

    if not weight_bytes:
        weight_bytes = safetensors_bytes

    if not has_safetensors:
        return (
            params_total,
            weight_bytes,
            False,
            "this checkpoint has no safetensors weights, which the serving loader requires",
        )
    if not weight_bytes:
        return params_total, weight_bytes, False, "the size of this checkpoint's weights could not be determined"

    return params_total, weight_bytes, True, ""


def _serving_facts(
    config: dict[str, Any],
    *,
    params_total: int,
    weight_bytes: int,
    streamable: bool,
    stream_blocked_reason: str,
    modality: tuple[str, ...],
    shard_config: ShardConfig | None,
) -> dict[str, Any]:
    """
    The `flyte.io/serving` blob: everything a serving backend needs to decide
    which GPUs this model fits on, and on which engines it can run at all.

    Derived from config.json, whose field names are the de-facto transformers
    schema. Consumers supply their own fallbacks for the two fields older
    configs routinely omit (`head_dim`, `num_key_value_heads`), so absent is
    represented as 0 rather than guessed at here -- a guess made in the producer
    is indistinguishable from a measurement once it is written down.
    """
    # Multimodal configs nest the language model's geometry, and the language
    # model is what dominates both the weights and the KV cache.
    text_config = config.get("text_config")
    text: dict[str, Any] = text_config if isinstance(text_config, dict) else {}

    def num(key: str) -> int:
        for candidate in (key, *_CONFIG_ALIASES.get(key, ())):
            value = config.get(candidate, text.get(candidate))
            if isinstance(value, (int, float)):
                return int(value)
        return 0

    facts: dict[str, Any] = {
        "v": SERVING_FACTS_VERSION,
        "params_total": params_total,
        "weight_bytes": weight_bytes,
        # transformers renamed this to `dtype`; older checkpoints still carry
        # `torch_dtype`, and plenty of live repos have only the old spelling.
        "torch_dtype": str(config.get("torch_dtype") or config.get("dtype") or text.get("torch_dtype") or ""),
        "num_hidden_layers": num("num_hidden_layers"),
        "num_attention_heads": num("num_attention_heads"),
        "num_key_value_heads": num("num_key_value_heads"),
        "head_dim": num("head_dim"),
        "hidden_size": num("hidden_size"),
        "vocab_size": num("vocab_size"),
        "max_position_embeddings": num("max_position_embeddings"),
        "architectures": list(config.get("architectures") or text.get("architectures") or []),
        "modality": list(modality),
        "streamable": streamable,
        "stream_blocked_reason": stream_blocked_reason,
    }

    quant = config.get("quantization_config")
    if isinstance(quant, dict):
        facts["quantization"] = {
            "method": str(quant.get("quant_method") or ""),
            "bits": int(quant.get("bits") or 0),
        }

    if shard_config is not None:
        # Recorded because it is irreversible: the loader reads exactly the
        # per-rank files sharding wrote, so this artifact is servable at this
        # engine and this degree, and nothing else.
        facts["sharding"] = {
            "engine": shard_config.engine,
            "tp": shard_config.args.tensor_parallel_size,
        }

    return facts


def _stream_to_remote_dir(
    repo_id: str,
    commit: str,
    token: str | None,
    remote_dir_path: str,
) -> tuple[str, str | None]:
    """
    Stream files directly from HuggingFace to a remote directory.

    Args:
        repo_id: The HuggingFace repository ID.
        commit: The commit ID.
        token: HuggingFace token.
        remote_dir_path: Path to the remote directory.

    Returns:
        Tuple of (remote_dir_path, readme_content).
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

    Args:
        repo_id: The HuggingFace repository ID.
        commit: The commit ID.
        token: HuggingFace token.
        local_dir: Local directory to download to.

    Returns:
        Tuple of (local_dir, readme_content).
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
    token: str | None,
    model_path: str,
    output_dir: str,
) -> tuple[str, str | None]:
    """
    Shard a model using vLLM.

    Args:
        shard_config: Sharding configuration.
        model_path: Path to the model to shard.
        output_dir: Directory to save sharded model.

    Returns:
        Path to sharded model directory.
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


def _wrap_as_model_artifact(
    result_dir: Dir,
    info: HuggingFaceModelInfo,
    artifact_name: str,
    commit: str,
    card_md: str | None,
    serving_facts: dict[str, Any] | None = None,
) -> Dir:
    """
    Wrap the stored model Dir with artifact metadata so the platform records a
    model artifact when the task succeeds: name is the artifact name, version is
    the HuggingFace commit (re-prefetching the same commit republishes the same
    version), and the repo README becomes the model card.
    """
    import flyte.artifacts as artifacts

    card = None
    if card_md:
        # HuggingFace READMEs open with a YAML frontmatter block (tags, license,
        # ...) that renders as literal text in a markdown card — drop it.
        content = re.sub(r"\A\s*---\n.*?\n---\n", "", card_md, count=1, flags=re.DOTALL) or card_md
        # Prefer an HTML card: the UI renders HTML cards in an iframe, which
        # works against any object store; markdown cards need a browser fetch
        # of the presigned URL and thus CORS on the bucket.
        fmt: str = "md"
        try:
            import markdown

            # "extra" bundles fenced code blocks and tables, which HF READMEs
            # use heavily (bibtex blocks, benchmark tables).
            content = _CARD_HTML_PREFIX + markdown.markdown(content, extensions=["extra"]) + _CARD_HTML_SUFFIX
            fmt = "html"
        except Exception as e:
            logger.warning(f"Markdown-to-HTML conversion unavailable, uploading markdown card: {e}")
        try:
            card = artifacts.Card.create_from(content=content, format=fmt, card_type="model")  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(f"Could not upload model card: {e}")

    attrs = {"source_repo": info.repo, "source_commit": commit}
    if info.shard_config is not None:
        attrs["sharding"] = f"{info.shard_config.engine}-tp{info.shard_config.args.tensor_parallel_size}"
    if serving_facts is not None:
        import json

        # Compact separators because this rides in a map<string,string> attr and
        # nothing reads it by eye.
        attrs[SERVING_ATTR_KEY] = json.dumps(serving_facts, separators=(",", ":"))

    metadata = artifacts.Metadata.create_model_metadata(
        name=artifact_name,
        version=commit,
        description=info.short_description or f"HuggingFace model {info.repo}",
        card=card,
        framework="huggingface",
        model_type=info.model_type,
        architecture=info.architecture,
        task=info.task,
        modality=info.modality,
        serial_format=info.serial_format or "safetensors",
        attrs=attrs,
    )
    return artifacts.new(result_dir, metadata)


# NOTE: the info argument is a json string instead of a HuggingFaceModelInfo
# object because the type engine cannot handle nested pydantic or dataclass
# objects when run in interactive mode.
def store_hf_model_task(info: str, raw_data_path: str | None = None) -> Dir:
    """Task to store a HuggingFace model."""

    import huggingface_hub

    import flyte.report

    # Get HF token from secrets; absent means anonymous access (public models).
    token = os.environ.get("HF_TOKEN") or None

    # Validate repo exists and get latest commit
    _info: HuggingFaceModelInfo = HuggingFaceModelInfo.model_validate_json(info)
    if not huggingface_hub.repo_exists(_info.repo, token=token):
        raise ValueError(f"Repository {_info.repo} does not exist in HuggingFace.")

    commit = huggingface_hub.list_repo_commits(_info.repo, token=token)[0].commit_id
    logger.info(f"Latest commit: {commit}")

    # Fetched unconditionally, even when the caller supplied model_type and
    # architecture: the serving facts are derived from the config regardless,
    # and it is a single ~2KB download.
    config: dict[str, Any] = {}
    logger.info("Looking up HuggingFace model info...")
    try:
        _model_type, _architecture, config = _lookup_huggingface_model_info(_info.repo, commit, token)
        _info.model_type = _info.model_type or _model_type
        _info.architecture = _info.architecture or _architecture
    except Exception as e:
        logger.warning(f"Warning: Could not lookup model info: {e}")
        _info.model_type = _info.model_type or "custom"
        _info.architecture = _info.architecture or "custom"

    logger.info(f"Model type: {_info.model_type}, architecture: {_info.architecture}")

    # Sizing metadata, read from the Hub's own records rather than from the
    # weights. Best-effort on purpose: a model that cannot be measured is still
    # a perfectly good artifact, so a failure here publishes without facts
    # rather than failing a download that may have taken an hour.
    serving_facts: dict[str, Any] | None = None
    try:
        params_total, weight_bytes, streamable, blocked_reason = _hf_weight_stats(_info.repo, commit, token)
        serving_facts = _serving_facts(
            config,
            params_total=params_total,
            weight_bytes=weight_bytes,
            streamable=streamable,
            stream_blocked_reason=blocked_reason,
            modality=_info.modality,
            shard_config=_info.shard_config,
        )
        logger.info(f"Serving facts: {weight_bytes} weight bytes, streamable={streamable}")
    except Exception as e:
        logger.warning(f"Could not derive serving facts for {_info.repo}: {e}")

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
                remote_path = flyte.ctx().raw_data_path.get_random_remote_path(artifact_name)

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

            report = markdown.markdown(card, extensions=["extra"])
        except Exception:
            report = card  # fallback to plain markdown content
        flyte.report.log(report)
        flyte.report.flush()

    logger.info(f"Model stored successfully at {result_dir.path}")
    return _wrap_as_model_artifact(result_dir, _info, artifact_name, commit, card, serving_facts)


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
    hf_token_key: str | None = "HF_TOKEN",
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

    On success the platform records a **model artifact** for the stored Dir:
    the artifact name is `artifact_name` (default: the repo name), the version
    is the HuggingFace commit id, the searchable metadata carries the model
    facts (framework/architecture/task/modality/serial_format plus the source
    repo and commit), and the repo's README, if it has one, is attached as the
    model card.
    Retrieve it later with `flyte.remote.Artifact.get(artifact_name)`.

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
        resources=flyte.Resources(cpu="8", memory="64Gi", disk="500Gi", gpu="A100:8"),
        hf_token_key="HF_TOKEN",
    )
    run.wait()
    ```

    Args:
        repo: The HuggingFace repository ID (e.g., 'meta-llama/Llama-2-7b-hf').
        artifact_name: Optional name for the stored artifact. If not provided,
            the repo name will be used (with '.' replaced by '-').
        architecture: Model architecture from HuggingFace config.json.
        task: Model task (e.g., 'generate', 'classify', 'embed'). Default: 'auto'.
        modality: Modalities supported by the model. Default: ('text',).
        serial_format: Model serialization format (e.g., 'safetensors', 'onnx').
        model_type: Model type (e.g., 'transformer', 'custom').
        short_description: Short description of the model.
        shard_config: Optional configuration for model sharding with vLLM.
        hf_token_key: Name of the secret containing the HuggingFace token. Default: 'HF_TOKEN'.
            Pass None to prefetch public models anonymously (no secret required).
        raw_data_path: Object store path to store the model. If not provided, the model is
            stored using the default path generated by the Flyte storage layer.
        resources: Resources for the prefetch task. Default:
            `Resources(cpu="2", memory="8Gi", disk="50Gi")`, which has no accelerator -- set
            this explicitly when sharding, and size `disk` for the model's weights. Accelerators
            go in `Resources(gpu=...)` as '{type}:{quantity}' (e.g., 'A100:8', 'L4:1').
            The `flyte prefetch hf-model` CLI exposes the same settings as separate
            `--cpu/--mem/--disk/--gpu/--shm` flags and folds them into this one argument.
        force: Force re-prefetch. Increment to force a new prefetch. Default: 0.

    Returns:
        A Run object representing the prefetch task execution.
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
        secrets=[Secret(key=hf_token_key, as_env_var="HF_TOKEN")] if hf_token_key else None,
    )
    prefetch_task = env.task(report=True, produces_artifacts=True)(store_hf_model_task)
    # Label the run with the model being prefetched so runs are searchable by model.
    model_label = artifact_name or repo.rsplit("/", maxsplit=1)[-1].replace(".", "-")
    run = flyte.with_runcontext(
        interactive_mode=True,
        disable_run_cache=disable_run_cache,
        labels={"model": model_label},
    ).run(prefetch_task, info.model_dump_json(), raw_data_path)
    return typing.cast(Run, run)
