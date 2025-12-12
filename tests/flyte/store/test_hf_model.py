"""Tests for flyte.store._hf_model module."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from flyte.store._hf_model import (
    HF_DOWNLOAD_IMAGE_PACKAGES,
    VLLM_SHARDING_IMAGE_PACKAGES,
    HuggingFaceModelInfo,
    ShardConfig,
    StoredModelInfo,
    VLLMShardArgs,
    _download_snapshot_to_local,
    _lookup_huggingface_model_info,
    _validate_artifact_name,
)

# =============================================================================
# VLLMShardArgs Tests
# =============================================================================


def test_vllm_shard_args_default_values():
    """Test default values are set correctly."""
    args = VLLMShardArgs()
    assert args.tensor_parallel_size == 1
    assert args.dtype == "auto"
    assert args.trust_remote_code is True
    assert args.max_model_len is None
    assert args.file_pattern == "*.safetensors"
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


@patch("huggingface_hub.hf_hub_download")
def test_lookup_huggingface_model_info_with_architectures_list(mock_download, tmp_path):
    """Test lookup when config has architectures list."""
    config_file = tmp_path / "config.json"
    config_data = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
    }
    config_file.write_text(json.dumps(config_data))
    mock_download.return_value = str(config_file)

    model_type, arch = _lookup_huggingface_model_info("meta-llama/Llama-2-7b-hf", "abc123", "token")

    assert model_type == "llama"
    assert arch == "LlamaForCausalLM"
    mock_download.assert_called_once_with(
        repo_id="meta-llama/Llama-2-7b-hf",
        filename="config.json",
        revision="abc123",
        token="token",
    )


@patch("huggingface_hub.hf_hub_download")
def test_lookup_huggingface_model_info_with_single_architecture(mock_download, tmp_path):
    """Test lookup when config has single architecture string."""
    config_file = tmp_path / "config.json"
    config_data = {
        "architecture": "GPT2LMHeadModel",
        "model_type": "gpt2",
    }
    config_file.write_text(json.dumps(config_data))
    mock_download.return_value = str(config_file)

    model_type, arch = _lookup_huggingface_model_info("gpt2", "main", None)

    assert model_type == "gpt2"
    assert arch == "GPT2LMHeadModel"


@patch("huggingface_hub.hf_hub_download")
def test_lookup_huggingface_model_info_with_multiple_architectures(mock_download, tmp_path):
    """Test lookup when config has multiple architectures."""
    config_file = tmp_path / "config.json"
    config_data = {
        "architectures": ["BertModel", "BertForMaskedLM"],
        "model_type": "bert",
    }
    config_file.write_text(json.dumps(config_data))
    mock_download.return_value = str(config_file)

    model_type, arch = _lookup_huggingface_model_info("bert-base", "main", None)

    assert model_type == "bert"
    assert arch == "BertModel,BertForMaskedLM"


@patch("huggingface_hub.hf_hub_download")
def test_lookup_huggingface_model_info_with_missing_fields(mock_download, tmp_path):
    """Test lookup when config is missing fields."""
    config_file = tmp_path / "config.json"
    config_data = {"hidden_size": 768}  # No architecture or model_type
    config_file.write_text(json.dumps(config_data))
    mock_download.return_value = str(config_file)

    model_type, arch = _lookup_huggingface_model_info("custom-model", "main", None)

    assert model_type is None
    assert arch is None


# =============================================================================
# _download_snapshot_to_local Tests
# =============================================================================


@patch("huggingface_hub.snapshot_download")
@patch("huggingface_hub.HfFileSystem")
def test_download_snapshot_to_local_with_readme(mock_hf_fs_class, mock_snapshot):
    """Test downloading snapshot with README."""
    mock_hfs = MagicMock()
    mock_hf_fs_class.return_value = mock_hfs

    # Mock README info
    mock_hfs.info.return_value = {"name": "repo/README.md"}

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
        mock_snapshot.assert_called_once_with(
            repo_id="test-repo",
            revision="abc123",
            local_dir=local_dir,
            token="token",
        )


@patch("huggingface_hub.snapshot_download")
@patch("huggingface_hub.HfFileSystem")
def test_download_snapshot_to_local_without_readme(mock_hf_fs_class, mock_snapshot):
    """Test downloading snapshot when README doesn't exist."""
    mock_hfs = MagicMock()
    mock_hf_fs_class.return_value = mock_hfs
    mock_hfs.info.side_effect = FileNotFoundError("No README")

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
    assert "vllm>=0.6.0" in VLLM_SHARDING_IMAGE_PACKAGES
    assert "markdown>=3.10" in VLLM_SHARDING_IMAGE_PACKAGES


# =============================================================================
# hf_model Function Tests
# =============================================================================


def test_hf_model_invalid_artifact_name_raises():
    """Test that invalid artifact name raises ValueError."""
    from flyte.store._hf_model import hf_model

    with pytest.raises(ValueError, match="must only contain alphanumeric characters"):
        hf_model(
            repo="meta-llama/Llama-2-7b-hf",
            artifact_name="invalid/name",
        )


def test_hf_model_invalid_accelerator_raises():
    """Test that invalid accelerator raises ValueError."""
    from flyte.store._hf_model import hf_model

    with pytest.raises(ValueError, match="Invalid accelerator"):
        hf_model(
            repo="meta-llama/Llama-2-7b-hf",
            accelerator="InvalidGPU:1",  # type: ignore
        )


# =============================================================================
# store_hf_model_task Tests
# =============================================================================


@patch("huggingface_hub.list_repo_commits")
@patch("huggingface_hub.repo_exists")
def test_store_hf_model_task_nonexistent_repo_raises(mock_repo_exists, mock_list_commits):
    """Test store task raises for non-existent repo."""
    from flyte.store._hf_model import store_hf_model_task

    mock_repo_exists.return_value = False

    info = HuggingFaceModelInfo(repo="nonexistent/model")

    with patch.dict(os.environ, {"HF_TOKEN": "test-token"}):
        with pytest.raises(ValueError, match="does not exist"):
            store_hf_model_task(info.model_dump_json())


# =============================================================================
# _shard_model Tests
# =============================================================================


def test_shard_model_invalid_engine():
    """Test that non-vllm engines raise an assertion error."""
    from flyte.store._hf_model import _shard_model

    # Create a ShardConfig with modified engine (bypassing Literal validation)
    shard_config = ShardConfig()
    object.__setattr__(shard_config, "engine", "invalid_engine")

    with pytest.raises(AssertionError, match="vllm"):
        _shard_model(
            repo="test/model",
            commit="abc123",
            shard_config=shard_config,
            token="token",
            model_path="/tmp/model",
            output_dir="/tmp/output",
        )
