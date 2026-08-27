"""Tests for flyte.prefetch._hf_model module."""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from flyte.prefetch._hf_model import (
    HF_DOWNLOAD_IMAGE_PACKAGES,
    VLLM_SHARDING_IMAGE_PACKAGES,
    VLLM_SHARDING_VERSION,
    HuggingFaceModelInfo,
    ShardConfig,
    StoredModelInfo,
    VLLMShardArgs,
    _validate_artifact_name,
)

# =============================================================================
# VLLMShardArgs Tests
# =============================================================================


def test_vllm_shard_args_default_values():
    """Test default values are set correctly."""
    from flyte.prefetch._hf_model import DEFAULT_SHARD_PATTERN

    args = VLLMShardArgs()
    assert args.tensor_parallel_size == 1
    assert args.dtype == "auto"
    assert args.trust_remote_code is True
    assert args.max_model_len is None
    assert args.file_pattern == DEFAULT_SHARD_PATTERN
    assert args.max_file_size == 5 * 1024**3  # 5GB


def test_vllm_shard_args_custom_values():
    """Test custom values are set correctly."""
    args = VLLMShardArgs(
        tensor_parallel_size=8,
        dtype="float16",
        trust_remote_code=False,
        max_model_len=4096,
        file_pattern="*.bin",
        max_file_size=10 * 1024**3,
    )
    assert args.tensor_parallel_size == 8
    assert args.dtype == "float16"
    assert args.trust_remote_code is False
    assert args.max_model_len == 4096
    assert args.file_pattern == "*.bin"
    assert args.max_file_size == 10 * 1024**3


def test_vllm_shard_args_get_vllm_args_basic():
    """Test get_vllm_args returns correct dictionary."""
    args = VLLMShardArgs(tensor_parallel_size=4)
    result = args.get_vllm_args("/path/to/model")

    assert result["model"] == "/path/to/model"
    assert result["tensor_parallel_size"] == 4
    assert result["dtype"] == "auto"
    assert result["trust_remote_code"] is True
    assert "max_model_len" not in result


def test_vllm_shard_args_get_vllm_args_with_max_model_len():
    """Test get_vllm_args includes max_model_len when set."""
    args = VLLMShardArgs(max_model_len=2048)
    result = args.get_vllm_args("/path/to/model")

    assert result["max_model_len"] == 2048


def test_vllm_shard_args_large_tensor_parallel_size():
    """Test VLLMShardArgs with large tensor_parallel_size."""
    args = VLLMShardArgs(tensor_parallel_size=16)
    vllm_args = args.get_vllm_args("/path/model")
    assert vllm_args["tensor_parallel_size"] == 16


def test_vllm_shard_args_different_dtype_values():
    """Test VLLMShardArgs with different dtype values."""
    for dtype in ["auto", "float16", "bfloat16", "float32"]:
        args = VLLMShardArgs(dtype=dtype)
        vllm_args = args.get_vllm_args("/path/model")
        assert vllm_args["dtype"] == dtype


def test_vllm_shard_args_custom_file_pattern():
    """Test VLLMShardArgs with custom file pattern."""
    args = VLLMShardArgs(file_pattern="model-*.safetensors")
    assert args.file_pattern == "model-*.safetensors"


def test_vllm_shard_args_custom_max_file_size():
    """Test VLLMShardArgs with custom max_file_size."""
    args = VLLMShardArgs(max_file_size=10 * 1024**3)  # 10GB
    assert args.max_file_size == 10 * 1024**3


# =============================================================================
# ShardConfig Tests
# =============================================================================


def test_shard_config_default_values():
    """Test default values are set correctly."""
    config = ShardConfig()
    assert config.engine == "vllm"
    assert isinstance(config.args, VLLMShardArgs)


def test_shard_config_custom_args():
    """Test custom args are set correctly."""
    custom_args = VLLMShardArgs(tensor_parallel_size=8)
    config = ShardConfig(args=custom_args)

    assert config.engine == "vllm"
    assert config.args.tensor_parallel_size == 8


# =============================================================================
# HuggingFaceModelInfo Tests
# =============================================================================


def test_huggingface_model_info_minimal_init():
    """Test initialization with only required field."""
    info = HuggingFaceModelInfo(repo="meta-llama/Llama-2-7b-hf")

    assert info.repo == "meta-llama/Llama-2-7b-hf"
    assert info.artifact_name is None
    assert info.architecture is None
    assert info.task == "auto"
    assert info.modality == ("text",)
    assert info.serial_format is None
    assert info.model_type is None
    assert info.short_description is None
    assert info.shard_config is None


def test_huggingface_model_info_full_init():
    """Test initialization with all fields."""
    shard_config = ShardConfig(args=VLLMShardArgs(tensor_parallel_size=4))
    info = HuggingFaceModelInfo(
        repo="meta-llama/Llama-2-7b-hf",
        artifact_name="llama-2-7b",
        architecture="LlamaForCausalLM",
        task="generate",
        modality=("text", "image"),
        serial_format="safetensors",
        model_type="llama",
        short_description="Llama 2 7B model",
        shard_config=shard_config,
    )

    assert info.repo == "meta-llama/Llama-2-7b-hf"
    assert info.artifact_name == "llama-2-7b"
    assert info.architecture == "LlamaForCausalLM"
    assert info.task == "generate"
    assert info.modality == ("text", "image")
    assert info.serial_format == "safetensors"
    assert info.model_type == "llama"
    assert info.short_description == "Llama 2 7B model"
    assert info.shard_config is not None
    assert info.shard_config.args.tensor_parallel_size == 4


def test_huggingface_model_info_model_dump():
    """Test HuggingFaceModelInfo can be serialized to dict."""
    info = HuggingFaceModelInfo(
        repo="meta-llama/Llama-2-7b-hf",
        artifact_name="llama-7b",
        task="generate",
    )

    dumped = info.model_dump()
    assert dumped["repo"] == "meta-llama/Llama-2-7b-hf"
    assert dumped["artifact_name"] == "llama-7b"
    assert dumped["task"] == "generate"


def test_huggingface_model_info_model_json():
    """Test HuggingFaceModelInfo can be serialized to JSON."""
    info = HuggingFaceModelInfo(
        repo="meta-llama/Llama-2-7b-hf",
        shard_config=ShardConfig(args=VLLMShardArgs(tensor_parallel_size=4)),
    )

    json_str = info.model_dump_json()
    assert "meta-llama/Llama-2-7b-hf" in json_str
    assert "tensor_parallel_size" in json_str


def test_huggingface_model_info_from_dict():
    """Test HuggingFaceModelInfo can be deserialized from dict."""
    data = {
        "repo": "meta-llama/Llama-2-7b-hf",
        "artifact_name": "llama-7b",
        "task": "generate",
        "modality": ("text",),
        "shard_config": {"engine": "vllm", "args": {"tensor_parallel_size": 8}},
    }

    info = HuggingFaceModelInfo(**data)
    assert info.repo == "meta-llama/Llama-2-7b-hf"
    assert info.shard_config.args.tensor_parallel_size == 8


# =============================================================================
# StoredModelInfo Tests
# =============================================================================


def test_stored_model_info_init():
    """Test initialization."""
    info = StoredModelInfo(
        artifact_name="my-model",
        path="s3://bucket/path/to/model",
        metadata={"version": "1.0", "format": "safetensors"},
    )

    assert info.artifact_name == "my-model"
    assert info.path == "s3://bucket/path/to/model"
    assert info.metadata == {"version": "1.0", "format": "safetensors"}


# =============================================================================
# _validate_artifact_name Tests
# =============================================================================


def test_validate_artifact_name_valid_names():
    """Test valid artifact names don't raise."""
    valid_names = [
        "my-model",
        "my_model",
        "MyModel",
        "model123",
        "Model-123_test",
        "ALLCAPS",
        "lowercase",
        "a",
        "1",
    ]
    for name in valid_names:
        _validate_artifact_name(name)  # Should not raise


def test_validate_artifact_name_none_is_valid():
    """Test None is accepted (will use default)."""
    _validate_artifact_name(None)  # Should not raise


def test_validate_artifact_name_invalid_names():
    """Test invalid artifact names raise ValueError."""
    invalid_names = [
        "my model",  # space
        "my.model",  # dot
        "my/model",  # slash
        "my:model",  # colon
        "my@model",  # at sign
        "my!model",  # exclamation
        "meta-llama/Llama-2-7b",  # slash
    ]
    for name in invalid_names:
        with pytest.raises(ValueError, match="must only contain alphanumeric characters"):
            _validate_artifact_name(name)


# =============================================================================
# _lookup_huggingface_model_info Tests
# =============================================================================


def test_lookup_huggingface_model_info_with_architectures_list(tmp_path):
    """Test lookup when config has architectures list."""
    config_file = tmp_path / "config.json"
    config_data = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
    }
    config_file.write_text(json.dumps(config_data))

    mock_hf_hub = MagicMock()
    mock_hf_hub.hf_hub_download.return_value = str(config_file)

    with patch.dict(sys.modules, {"huggingface_hub": mock_hf_hub}):
        from flyte.prefetch._hf_model import _lookup_huggingface_model_info

        model_type, arch, config = _lookup_huggingface_model_info("meta-llama/Llama-2-7b-hf", "abc123", "token")

        assert model_type == "llama"
        assert arch == "LlamaForCausalLM"
        assert config == config_data
        mock_hf_hub.hf_hub_download.assert_called_once_with(
            repo_id="meta-llama/Llama-2-7b-hf",
            filename="config.json",
            revision="abc123",
            token="token",
        )


def test_lookup_huggingface_model_info_with_single_architecture(tmp_path):
    """Test lookup when config has single architecture string."""
    config_file = tmp_path / "config.json"
    config_data = {
        "architecture": "GPT2LMHeadModel",
        "model_type": "gpt2",
    }
    config_file.write_text(json.dumps(config_data))

    mock_hf_hub = MagicMock()
    mock_hf_hub.hf_hub_download.return_value = str(config_file)

    with patch.dict(sys.modules, {"huggingface_hub": mock_hf_hub}):
        from flyte.prefetch._hf_model import _lookup_huggingface_model_info

        model_type, arch, config = _lookup_huggingface_model_info("gpt2", "main", None)

        assert model_type == "gpt2"
        assert arch == "GPT2LMHeadModel"
        assert config == config_data


def test_lookup_huggingface_model_info_with_multiple_architectures(tmp_path):
    """Test lookup when config has multiple architectures."""
    config_file = tmp_path / "config.json"
    config_data = {
        "architectures": ["BertModel", "BertForMaskedLM"],
        "model_type": "bert",
    }
    config_file.write_text(json.dumps(config_data))

    mock_hf_hub = MagicMock()
    mock_hf_hub.hf_hub_download.return_value = str(config_file)

    with patch.dict(sys.modules, {"huggingface_hub": mock_hf_hub}):
        from flyte.prefetch._hf_model import _lookup_huggingface_model_info

        model_type, arch, config = _lookup_huggingface_model_info("bert-base", "main", None)

        assert model_type == "bert"
        assert arch == "BertModel,BertForMaskedLM"
        assert config == config_data


def test_lookup_huggingface_model_info_with_missing_fields(tmp_path):
    """Test lookup when config is missing fields."""
    config_file = tmp_path / "config.json"
    config_data = {"hidden_size": 768}  # No architecture or model_type
    config_file.write_text(json.dumps(config_data))

    mock_hf_hub = MagicMock()
    mock_hf_hub.hf_hub_download.return_value = str(config_file)

    with patch.dict(sys.modules, {"huggingface_hub": mock_hf_hub}):
        from flyte.prefetch._hf_model import _lookup_huggingface_model_info

        model_type, arch, config = _lookup_huggingface_model_info("custom-model", "main", None)

        assert model_type is None
        assert arch is None
        # The raw config is returned even when the summary fields are absent --
        # it is what the serving facts are derived from.
        assert config == config_data


# =============================================================================
# _download_snapshot_to_local Tests
# =============================================================================


def test_download_snapshot_to_local_with_readme():
    """Test downloading snapshot with README."""
    mock_hf_hub = MagicMock()
    mock_hfs = MagicMock()
    mock_hf_hub.HfFileSystem.return_value = mock_hfs

    # Mock README info
    mock_hfs.info.return_value = {"name": "repo/README.md"}

    with patch.dict(sys.modules, {"huggingface_hub": mock_hf_hub}):
        from flyte.prefetch._hf_model import _download_snapshot_to_local

        with tempfile.TemporaryDirectory() as local_dir:
            with patch("tempfile.NamedTemporaryFile") as mock_temp:
                mock_temp_file = MagicMock()
                mock_temp_file.name = "/tmp/readme"
                mock_temp.return_value.__enter__.return_value = mock_temp_file

                with patch(
                    "builtins.open",
                    MagicMock(
                        return_value=MagicMock(
                            __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value="# README"))),
                            __exit__=MagicMock(),
                        )
                    ),
                ):
                    result_dir, card = _download_snapshot_to_local("test-repo", "abc123", "token", local_dir)

            assert result_dir == local_dir
            assert card is not None
            mock_hf_hub.snapshot_download.assert_called_once_with(
                repo_id="test-repo",
                revision="abc123",
                local_dir=local_dir,
                token="token",
            )


def test_download_snapshot_to_local_without_readme():
    """Test downloading snapshot when README doesn't exist."""
    mock_hf_hub = MagicMock()
    mock_hfs = MagicMock()
    mock_hf_hub.HfFileSystem.return_value = mock_hfs
    mock_hfs.info.side_effect = FileNotFoundError("No README")

    with patch.dict(sys.modules, {"huggingface_hub": mock_hf_hub}):
        from flyte.prefetch._hf_model import _download_snapshot_to_local

        with tempfile.TemporaryDirectory() as local_dir:
            result_dir, card = _download_snapshot_to_local("test-repo", "main", None, local_dir)

        assert result_dir == local_dir
        assert card is None


# =============================================================================
# Image Package Constants Tests
# =============================================================================


def test_hf_download_image_packages():
    """Test HF download image packages are defined."""
    assert "huggingface-hub>=0.27.0" in HF_DOWNLOAD_IMAGE_PACKAGES
    assert "hf-transfer>=0.1.8" in HF_DOWNLOAD_IMAGE_PACKAGES
    assert "markdown>=3.10" in HF_DOWNLOAD_IMAGE_PACKAGES


def test_vllm_sharding_image_packages():
    """Test vLLM sharding image packages are defined."""
    assert "huggingface-hub>=0.27.0" in VLLM_SHARDING_IMAGE_PACKAGES
    assert "hf-transfer>=0.1.8" in VLLM_SHARDING_IMAGE_PACKAGES
    assert f"vllm=={VLLM_SHARDING_VERSION}" in VLLM_SHARDING_IMAGE_PACKAGES
    assert "markdown>=3.10" in VLLM_SHARDING_IMAGE_PACKAGES


def test_vllm_sharding_pin_is_exact():
    """The sharding vLLM is pinned, not floored.

    A floor lets the resolver pick whatever is newest at image-build time, which silently
    decouples the vLLM writing the sharded state from the one reading it back at serve time.
    """
    vllm_specs = [pkg for pkg in VLLM_SHARDING_IMAGE_PACKAGES if pkg.startswith("vllm")]
    assert vllm_specs == [f"vllm=={VLLM_SHARDING_VERSION}"]


# =============================================================================
# hf_model Function Tests
# =============================================================================


def test_hf_model_invalid_artifact_name_raises():
    """Test that invalid artifact name raises ValueError."""
    from flyte.prefetch._hf_model import hf_model

    with pytest.raises(ValueError, match="must only contain alphanumeric characters"):
        hf_model(
            repo="meta-llama/Llama-2-7b-hf",
            artifact_name="invalid/name",
        )


def test_hf_model_invalid_gpu_raises():
    """Test that invalid gpu accelerator raises ValueError."""
    from flyte._resources import Resources
    from flyte.prefetch._hf_model import hf_model

    with pytest.raises(ValueError, match="gpu"):
        hf_model(
            repo="meta-llama/Llama-2-7b-hf",
            resources=Resources(gpu="InvalidGPU:1"),  # type: ignore
        )


# =============================================================================
# prefetch_hf_model_task Tests
# =============================================================================


def test_prefetch_hf_model_task_nonexistent_repo_raises():
    """Test prefetch task raises for non-existent repo."""
    from flyte.prefetch._hf_model import store_hf_model_task

    mock_hf_hub = MagicMock()
    mock_hf_hub.repo_exists.return_value = False

    with patch.dict(sys.modules, {"huggingface_hub": mock_hf_hub}):
        from flyte.prefetch._hf_model import store_hf_model_task

        info = HuggingFaceModelInfo(repo="nonexistent/model")

        with patch.dict(os.environ, {"HF_TOKEN": "test-token"}):
            with pytest.raises(ValueError, match="does not exist"):
                store_hf_model_task(info.model_dump_json())


# =============================================================================
# _wrap_as_model_artifact Tests
# =============================================================================


def test_wrap_as_model_artifact_metadata():
    """The stored Dir is wrapped with model artifact metadata: name, HF commit
    as version, model facts + source repo/commit as attrs, README as card."""
    from flyte.io import Dir
    from flyte.prefetch._hf_model import _wrap_as_model_artifact

    info = HuggingFaceModelInfo(
        repo="meta-llama/Llama-2-7b-hf",
        architecture="LlamaForCausalLM",
        model_type="llama",
        task="generate",
        short_description="Llama 2 7B",
    )
    result_dir = Dir(path="s3://bucket/models/llama")

    import flyte.artifacts as artifacts

    fake_card = artifacts.Card(uri="s3://b/model.md", format="md", card_type="model")
    with patch("flyte.artifacts.Card.create_from", return_value=fake_card) as mock_card:
        wrapped = _wrap_as_model_artifact(result_dir, info, "Llama-2-7b-hf", "abc123", "# Llama 2")

    kwargs = mock_card.call_args.kwargs
    assert kwargs["card_type"] == "model"
    # html when the markdown package is importable, md otherwise.
    assert kwargs["format"] in ("md", "html")
    assert "Llama 2" in kwargs["content"]
    md = wrapped.get_flyte_metadata()
    assert md.name == "Llama-2-7b-hf"
    assert md.version == "abc123"
    assert md.description == "Llama 2 7B"
    assert md.card == fake_card
    assert md.attrs["architecture"] == "LlamaForCausalLM"
    assert md.attrs["model_type"] == "llama"
    assert md.attrs["task"] == "generate"
    assert md.attrs["framework"] == "huggingface"
    assert md.attrs["serial_format"] == "safetensors"
    assert md.attrs["source_repo"] == "meta-llama/Llama-2-7b-hf"
    assert md.attrs["source_commit"] == "abc123"
    assert "sharding" not in md.attrs
    # The wrapper preserves the Dir interface.
    assert wrapped.path == "s3://bucket/models/llama"


def test_wrap_as_model_artifact_no_readme_no_card():
    """No README -> no card, and the description falls back to the repo."""
    from flyte.io import Dir
    from flyte.prefetch._hf_model import _wrap_as_model_artifact

    info = HuggingFaceModelInfo(repo="org/model")
    wrapped = _wrap_as_model_artifact(Dir(path="s3://b/m"), info, "model", "deadbeef", None)

    md = wrapped.get_flyte_metadata()
    assert md.card is None
    assert md.description == "HuggingFace model org/model"


def test_wrap_as_model_artifact_sharded_records_sharding():
    """Sharded prefetch records the sharding engine and parallelism."""
    from flyte.io import Dir
    from flyte.prefetch._hf_model import _wrap_as_model_artifact

    info = HuggingFaceModelInfo(
        repo="org/model",
        shard_config=ShardConfig(args=VLLMShardArgs(tensor_parallel_size=8)),
    )
    wrapped = _wrap_as_model_artifact(Dir(path="s3://b/m"), info, "model", "c1", None)
    assert wrapped.get_flyte_metadata().attrs["sharding"] == "vllm-tp8"


def test_wrap_as_model_artifact_strips_readme_frontmatter():
    """HF README YAML frontmatter is dropped from the uploaded card."""
    from flyte.io import Dir
    from flyte.prefetch._hf_model import _wrap_as_model_artifact

    readme = "---\nlanguage:\n  - en\nlicense: mit\n---\n# Model\nBody text."
    info = HuggingFaceModelInfo(repo="org/model")
    with patch("flyte.artifacts.Card.create_from") as mock_card:
        _wrap_as_model_artifact(Dir(path="s3://b/m"), info, "model", "c1", readme)

    content = mock_card.call_args.kwargs["content"]
    assert "license: mit" not in content
    assert "Body text." in content


def test_wrap_as_model_artifact_card_is_html_when_markdown_available():
    """With the markdown package present the card uploads as a rendered HTML page."""
    from flyte.io import Dir
    from flyte.prefetch._hf_model import _wrap_as_model_artifact

    pytest.importorskip("markdown")
    info = HuggingFaceModelInfo(repo="org/model")
    with patch("flyte.artifacts.Card.create_from") as mock_card:
        _wrap_as_model_artifact(Dir(path="s3://b/m"), info, "model", "c1", "# Title\nBody")

    assert mock_card.call_args.kwargs["format"] == "html"
    content = mock_card.call_args.kwargs["content"]
    assert content.startswith("<!DOCTYPE html>")
    assert "<h1>Title</h1>" in content


def test_wrap_as_model_artifact_card_upload_failure_is_nonfatal():
    """A card upload failure must not fail the prefetch."""
    from flyte.io import Dir
    from flyte.prefetch._hf_model import _wrap_as_model_artifact

    info = HuggingFaceModelInfo(repo="org/model")
    with patch("flyte.artifacts.Card.create_from", side_effect=RuntimeError("no storage")):
        wrapped = _wrap_as_model_artifact(Dir(path="s3://b/m"), info, "model", "c1", "# README")

    assert wrapped.get_flyte_metadata().card is None


# =============================================================================
# _shard_model Tests
# =============================================================================


def test_shard_model_valid_engine(tmp_path):
    """Test that valid vllm engines do not raise an assertion error."""
    from flyte.prefetch._hf_model import _shard_model

    shard_config = ShardConfig(engine="vllm")
    with patch.dict(sys.modules, {"vllm": MagicMock(), "huggingface_hub": MagicMock()}):
        from flyte.prefetch._hf_model import _shard_model

        _shard_model(
            repo="test/model",
            commit="abc123",
            shard_config=shard_config,
            token="token",
            model_path=str(tmp_path),
            output_dir=str(tmp_path),
        )


def test_shard_model_invalid_engine():
    """Test that non-vllm engines raise an assertion error."""
    from flyte.prefetch._hf_model import _shard_model

    # Create a ShardConfig with modified engine (bypassing Literal validation)
    shard_config = ShardConfig()
    object.__setattr__(shard_config, "engine", "invalid_engine")

    with patch.dict(sys.modules, {"vllm": MagicMock(), "huggingface_hub": MagicMock()}):
        from flyte.prefetch._hf_model import _shard_model

        with pytest.raises(AssertionError, match="vllm"):
            _shard_model(
                repo="test/model",
                commit="abc123",
                shard_config=shard_config,
                token="token",
                model_path="/tmp/model",
                output_dir="/tmp/output",
            )


# =============================================================================
# Serving facts Tests
# =============================================================================


def _facts(config, **overrides):
    """Call _serving_facts with the boring arguments defaulted."""
    from flyte.prefetch._hf_model import _serving_facts

    kwargs = {
        "params_total": 0,
        "weight_bytes": 0,
        "streamable": True,
        "stream_blocked_reason": "",
        "modality": ("text",),
        "shard_config": None,
    }
    kwargs.update(overrides)
    return _serving_facts(config, **kwargs)


def test_serving_facts_modern_config():
    """A current-generation config populates every geometry field."""
    from flyte.prefetch._hf_model import SERVING_FACTS_VERSION

    facts = _facts(
        {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "torch_dtype": "bfloat16",
            "num_hidden_layers": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "hidden_size": 5120,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
        },
        params_total=32_762_123_264,
        weight_bytes=65_524_246_528,
    )

    assert facts["v"] == SERVING_FACTS_VERSION
    assert facts["params_total"] == 32_762_123_264
    assert facts["weight_bytes"] == 65_524_246_528
    assert facts["torch_dtype"] == "bfloat16"
    assert facts["num_hidden_layers"] == 64
    assert facts["num_key_value_heads"] == 8
    assert facts["head_dim"] == 128
    assert facts["hidden_size"] == 5120
    assert facts["max_position_embeddings"] == 40960
    assert facts["streamable"] is True
    assert facts["stream_blocked_reason"] == ""
    assert facts["modality"] == ["text"]


def test_serving_facts_architectures_stays_a_list():
    """
    architectures is a list, not a comma-joined scalar.

    Joining it would deserialize as one bogus architecture name and fail every
    engine-support lookup on the consumer side.
    """
    facts = _facts({"architectures": ["BertModel", "BertForMaskedLM"]})

    assert facts["architectures"] == ["BertModel", "BertForMaskedLM"]


def test_serving_facts_legacy_gpt2_aliases():
    """
    GPT-2-family configs spell their geometry n_layer/n_head/n_embd.

    transformers reconciles these through each config class's attribute_map, but
    we read the raw JSON and so see the original names. A missed alias silently
    zeroes the KV-cache term, which under-estimates VRAM -- the direction that
    OOMs a deploy.
    """
    facts = _facts(
        {
            "architectures": ["GPT2LMHeadModel"],
            "model_type": "gpt2",
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "n_positions": 1024,
            "vocab_size": 50257,
        }
    )

    assert facts["num_hidden_layers"] == 12
    assert facts["num_attention_heads"] == 12
    assert facts["hidden_size"] == 768
    assert facts["max_position_embeddings"] == 1024
    assert facts["vocab_size"] == 50257


def test_serving_facts_absent_fields_are_zero_not_guessed():
    """
    head_dim and num_key_value_heads are routinely omitted by older configs.

    They are reported as 0 rather than derived here, because a guess made in the
    producer is indistinguishable from a measurement once written down -- the
    consumer supplies its own fallbacks.
    """
    facts = _facts({"num_hidden_layers": 12, "num_attention_heads": 12, "hidden_size": 768})

    assert facts["head_dim"] == 0
    assert facts["num_key_value_heads"] == 0


def test_serving_facts_reads_nested_text_config():
    """Multimodal configs nest the language model's geometry, which dominates."""
    facts = _facts(
        {
            "architectures": ["Llama4ForConditionalGeneration"],
            "text_config": {
                "num_hidden_layers": 48,
                "num_attention_heads": 40,
                "hidden_size": 5120,
                "torch_dtype": "bfloat16",
            },
        },
        modality=("text", "image"),
    )

    assert facts["num_hidden_layers"] == 48
    assert facts["num_attention_heads"] == 40
    assert facts["hidden_size"] == 5120
    assert facts["torch_dtype"] == "bfloat16"
    assert facts["modality"] == ["text", "image"]


def test_serving_facts_dtype_falls_back_to_new_spelling():
    """transformers renamed torch_dtype to dtype; older repos have only the old one."""
    assert _facts({"dtype": "float16"})["torch_dtype"] == "float16"
    assert _facts({"torch_dtype": "bfloat16"})["torch_dtype"] == "bfloat16"
    assert _facts({})["torch_dtype"] == ""


def test_serving_facts_quantization_present_and_absent():
    """The quantization sub-object appears only when the config declares one."""
    facts = _facts({"quantization_config": {"quant_method": "fp8", "bits": 8}})
    assert facts["quantization"] == {"method": "fp8", "bits": 8}

    assert "quantization" not in _facts({})


def test_serving_facts_records_sharding():
    """
    Sharding is recorded because it is irreversible: the loader reads exactly the
    per-rank files sharding wrote, so the artifact is servable at this engine and
    this degree and nothing else.
    """
    facts = _facts({}, shard_config=ShardConfig(args=VLLMShardArgs(tensor_parallel_size=8)))

    assert facts["sharding"] == {"engine": "vllm", "tp": 8}
    assert "sharding" not in _facts({})


def test_serving_facts_carries_stream_blocked_reason():
    """An unstreamable checkpoint carries the reason through to the consumer."""
    facts = _facts({}, streamable=False, stream_blocked_reason="no safetensors")

    assert facts["streamable"] is False
    assert facts["stream_blocked_reason"] == "no safetensors"


# =============================================================================
# _hf_weight_stats Tests
# =============================================================================


def _hf_hub_mock(*, scan=None, files=None, ls_raises=None, model_info_raises=None):
    """A huggingface_hub double exposing just model_info and HfFileSystem.ls."""
    from types import SimpleNamespace

    hub = MagicMock()
    if model_info_raises is not None:
        hub.HfApi.return_value.model_info.side_effect = model_info_raises
    else:
        hub.HfApi.return_value.model_info.return_value = SimpleNamespace(safetensors=scan)
    if ls_raises is not None:
        hub.HfFileSystem.return_value.ls.side_effect = ls_raises
    else:
        hub.HfFileSystem.return_value.ls.return_value = files or []
    return hub


def _file(name, size):
    return {"type": "file", "name": name, "size": size}


def test_hf_weight_stats_converts_scan_parameters_to_bytes():
    """
    The Hub's safetensors scan reports parameter counts per dtype, which converts
    to bytes exactly and sidesteps the double-counting that summing blobs hits on
    repos shipping both .bin and .safetensors.
    """
    from types import SimpleNamespace

    from flyte.prefetch._hf_model import _hf_weight_stats

    hub = _hf_hub_mock(
        scan=SimpleNamespace(total=1000, parameters={"BF16": 1000}),
        files=[_file("m/model.safetensors", 999)],
    )
    with patch.dict(sys.modules, {"huggingface_hub": hub}):
        params, weight_bytes, streamable, reason = _hf_weight_stats("m", "abc", None)

    assert params == 1000
    # bf16 is 2 bytes/param, and the scan wins over the 999-byte file listing.
    assert weight_bytes == 2000
    assert streamable is True
    assert reason == ""


def test_hf_weight_stats_unknown_dtype_falls_back_to_two_bytes():
    """Unrecognised dtypes assume 16-bit -- conservative rather than absent."""
    from types import SimpleNamespace

    from flyte.prefetch._hf_model import _hf_weight_stats

    hub = _hf_hub_mock(
        scan=SimpleNamespace(total=100, parameters={"some_future_dtype": 100}),
        files=[_file("m/model.safetensors", 1)],
    )
    with patch.dict(sys.modules, {"huggingface_hub": hub}):
        _, weight_bytes, _, _ = _hf_weight_stats("m", "abc", None)

    assert weight_bytes == 200


def test_hf_weight_stats_falls_back_to_file_sizes_without_a_scan():
    """Without a scan, the summed .safetensors blob sizes stand in."""
    from flyte.prefetch._hf_model import _hf_weight_stats

    hub = _hf_hub_mock(
        scan=None,
        files=[
            _file("m/model-00001-of-00002.safetensors", 500),
            _file("m/model-00002-of-00002.safetensors", 300),
            _file("m/README.md", 10),
        ],
    )
    with patch.dict(sys.modules, {"huggingface_hub": hub}):
        params, weight_bytes, streamable, reason = _hf_weight_stats("m", "abc", None)

    assert params == 0
    # Non-safetensors files do not count toward the weights.
    assert weight_bytes == 800
    assert streamable is True
    assert reason == ""


def test_hf_weight_stats_without_safetensors_is_not_streamable():
    """The serving loader reads safetensors and nothing else."""
    from flyte.prefetch._hf_model import _hf_weight_stats

    hub = _hf_hub_mock(scan=None, files=[_file("m/pytorch_model.bin", 1000)])
    with patch.dict(sys.modules, {"huggingface_hub": hub}):
        _, _, streamable, reason = _hf_weight_stats("m", "abc", None)

    assert streamable is False
    assert "safetensors" in reason


def test_hf_weight_stats_listing_failure_is_not_fatal():
    """
    A failed listing cannot prove the checkpoint unstreamable, but refusing to
    publish over it would be worse than publishing without facts.
    """
    from flyte.prefetch._hf_model import _hf_weight_stats

    hub = _hf_hub_mock(scan=None, ls_raises=RuntimeError("hub is down"))
    with patch.dict(sys.modules, {"huggingface_hub": hub}):
        _, _, streamable, reason = _hf_weight_stats("m", "abc", None)

    assert streamable is False
    assert "could not be listed" in reason


def test_hf_weight_stats_survives_scan_failure():
    """An unavailable scan degrades to the file listing rather than raising."""
    from flyte.prefetch._hf_model import _hf_weight_stats

    hub = _hf_hub_mock(
        model_info_raises=RuntimeError("gated repo"),
        files=[_file("m/model.safetensors", 750)],
    )
    with patch.dict(sys.modules, {"huggingface_hub": hub}):
        params, weight_bytes, streamable, _ = _hf_weight_stats("m", "abc", None)

    assert params == 0
    assert weight_bytes == 750
    assert streamable is True


# =============================================================================
# _wrap_as_model_artifact Tests
# =============================================================================


def test_wrap_as_model_artifact_writes_serving_facts_as_one_attr():
    """
    The facts ride as a single JSON value under one key, not as ~15 flat attrs.

    attrs is a map<string,string>, so spreading the schema across the key set
    would freeze it -- and it multiplies the index entries the control plane
    keeps per artifact.
    """
    import flyte.artifacts as artifacts
    from flyte.prefetch._hf_model import SERVING_ATTR_KEY, _wrap_as_model_artifact

    captured = {}
    facts = {"v": 1, "params_total": 751_632_384, "streamable": True}
    info = HuggingFaceModelInfo(repo="Qwen/Qwen3-0.6B")

    with (
        patch.object(artifacts.Metadata, "create_model_metadata", side_effect=lambda **kw: captured.update(kw)),
        patch.object(artifacts, "new", return_value="wrapped") as mock_new,
    ):
        result = _wrap_as_model_artifact(MagicMock(), info, "Qwen3-0-6B", "abc123", None, facts)

    assert result == "wrapped"
    assert mock_new.call_count == 1

    attrs = captured["attrs"]
    assert set(attrs) == {"source_repo", "source_commit", SERVING_ATTR_KEY}
    assert attrs["source_repo"] == "Qwen/Qwen3-0.6B"
    assert attrs["source_commit"] == "abc123"
    assert json.loads(attrs[SERVING_ATTR_KEY]) == facts
    # Compact separators: nothing reads this by eye.
    assert ", " not in attrs[SERVING_ATTR_KEY]
    assert ": " not in attrs[SERVING_ATTR_KEY]


def test_wrap_as_model_artifact_without_facts_omits_the_key():
    """A model that could not be measured is still a perfectly good artifact."""
    import flyte.artifacts as artifacts
    from flyte.prefetch._hf_model import SERVING_ATTR_KEY, _wrap_as_model_artifact

    captured = {}
    info = HuggingFaceModelInfo(repo="Qwen/Qwen3-0.6B")

    with (
        patch.object(artifacts.Metadata, "create_model_metadata", side_effect=lambda **kw: captured.update(kw)),
        patch.object(artifacts, "new", return_value="wrapped"),
    ):
        _wrap_as_model_artifact(MagicMock(), info, "Qwen3-0-6B", "abc123", None, None)

    assert SERVING_ATTR_KEY not in captured["attrs"]


def test_wrap_as_model_artifact_sharded_carries_both_attrs():
    """Sharding is recorded as its own flat attr as well as inside the facts."""
    import flyte.artifacts as artifacts
    from flyte.prefetch._hf_model import SERVING_ATTR_KEY, _wrap_as_model_artifact

    captured = {}
    info = HuggingFaceModelInfo(
        repo="Qwen/Qwen3-32B",
        shard_config=ShardConfig(args=VLLMShardArgs(tensor_parallel_size=8)),
    )

    with (
        patch.object(artifacts.Metadata, "create_model_metadata", side_effect=lambda **kw: captured.update(kw)),
        patch.object(artifacts, "new", return_value="wrapped"),
    ):
        _wrap_as_model_artifact(MagicMock(), info, "Qwen3-32B", "abc123", None, {"v": 1})

    attrs = captured["attrs"]
    assert attrs["sharding"] == "vllm-tp8"
    assert set(attrs) == {"source_repo", "source_commit", "sharding", SERVING_ATTR_KEY}
