"""Tests for flyte.cli._prefetch module."""

import tempfile
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from flyte.cli._prefetch import prefetch


@pytest.fixture
def runner():
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def mock_cfg():
    """Create a mock config object."""
    cfg = MagicMock()
    cfg.config.task.project = "default"
    cfg.config.task.domain = "development"
    return cfg


# =============================================================================
# Prefetch Group Tests
# =============================================================================


def test_prefetch_help(runner):
    """Test prefetch command group help."""
    result = runner.invoke(prefetch, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Prefetch artifacts from remote registries" in result.output


def test_prefetch_hf_model_subcommand_exists(runner):
    """Test hf-model subcommand is available."""
    result = runner.invoke(prefetch, ["--help"])
    assert result.exit_code == 0, result.output
    assert "hf-model" in result.output


# =============================================================================
# hf-model Command Tests
# =============================================================================


def test_hf_model_help(runner):
    """Test hf-model command help."""
    result = runner.invoke(prefetch, ["hf-model", "--help"])
    assert result.exit_code == 0, result.output
    assert "Prefetch a HuggingFace model" in result.output


def test_hf_model_help_shows_all_options(runner):
    """Test hf-model help shows all expected options."""
    result = runner.invoke(prefetch, ["hf-model", "--help"])
    assert result.exit_code == 0, result.output

    expected_options = [
        "--artifact-name",
        "--architecture",
        "--task",
        "--modality",
        "--format",
        "--model-type",
        "--short-description",
        "--force",
        "--wait",
        "--hf-token-key",
        "--cpu",
        "--mem",
        "--ephemeral-storage",
        "--accelerator",
        "--shard-config",
        "--project",
        "--domain",
    ]
    for option in expected_options:
        assert option in result.output, f"Option {option} not found in help"


def test_hf_model_requires_repo_argument(runner):
    """Test hf-model command requires repo argument."""
    result = runner.invoke(prefetch, ["hf-model"])
    assert result.exit_code != 0
    assert "Missing argument" in result.output or "REPO" in result.output


def test_hf_model_invalid_accelerator(runner, mock_cfg):
    """Test hf-model command rejects invalid accelerator."""
    result = runner.invoke(
        prefetch,
        [
            "hf-model",
            "meta-llama/Llama-2-7b-hf",
            "--accelerator",
            "InvalidGPU:1",
        ],
        obj=mock_cfg,
    )

    assert result.exit_code != 0
    # Click should reject invalid choice


def test_hf_model_shard_config_file_not_found(runner, mock_cfg):
    """Test hf-model command fails with non-existent shard config file."""
    result = runner.invoke(
        prefetch,
        [
            "hf-model",
            "meta-llama/Llama-2-7b-hf",
            "--shard-config",
            "/nonexistent/path/config.yaml",
        ],
        obj=mock_cfg,
    )

    assert result.exit_code != 0
    # Click should reject non-existent path


# =============================================================================
# Shard Config Parsing Tests
# =============================================================================


@patch("flyte.prefetch.hf_model")
@patch("flyte.cli._run.initialize_config")
def test_shard_config_parsing(mock_init_config, mock_prefetch, runner, mock_cfg):
    """Test parsing shard config file."""
    mock_init_config.return_value = mock_cfg

    mock_run = MagicMock()
    mock_run.url = "https://console.example.com/run/123"
    mock_prefetch.return_value = mock_run

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
engine: vllm
args:
  tensor_parallel_size: 8
  dtype: float16
  trust_remote_code: true
  max_model_len: 4096
""")
        shard_config_path = f.name

    try:
        result = runner.invoke(
            prefetch,
            [
                "hf-model",
                "meta-llama/Llama-2-70b-hf",
                "--shard-config",
                shard_config_path,
                "--accelerator",
                "A100:8",
            ],
            obj=mock_cfg,
        )

        assert result.exit_code == 0, result.output
        mock_prefetch.assert_called_once()
        call_kwargs = mock_prefetch.call_args.kwargs
        assert call_kwargs["shard_config"] is not None
        assert call_kwargs["shard_config"].engine == "vllm"
        assert call_kwargs["shard_config"].args.tensor_parallel_size == 8
        assert call_kwargs["shard_config"].args.dtype == "float16"
        assert call_kwargs["shard_config"].args.max_model_len == 4096
    finally:
        import os

        os.unlink(shard_config_path)


@patch("flyte.prefetch.hf_model")
@patch("flyte.cli._run.initialize_config")
def test_shard_config_minimal(mock_init_config, mock_prefetch, runner, mock_cfg):
    """Test parsing minimal shard config file."""
    mock_init_config.return_value = mock_cfg

    mock_run = MagicMock()
    mock_run.url = "https://console.example.com/run/123"
    mock_prefetch.return_value = mock_run

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
engine: vllm
args: {}
""")
        shard_config_path = f.name

    try:
        result = runner.invoke(
            prefetch,
            [
                "hf-model",
                "meta-llama/Llama-2-70b-hf",
                "--shard-config",
                shard_config_path,
            ],
            obj=mock_cfg,
        )

        assert result.exit_code == 0, result.output
        call_kwargs = mock_prefetch.call_args.kwargs
        assert call_kwargs["shard_config"] is not None
        assert call_kwargs["shard_config"].engine == "vllm"
        # Should use defaults for args
        assert call_kwargs["shard_config"].args.tensor_parallel_size == 1
    finally:
        import os

        os.unlink(shard_config_path)


# =============================================================================
# Basic Invocation Tests
# =============================================================================


@patch("flyte.prefetch.hf_model")
@patch("flyte.cli._run.initialize_config")
def test_hf_model_basic_call(mock_init_config, mock_prefetch, runner, mock_cfg):
    """Test basic hf-model command invocation."""
    mock_init_config.return_value = mock_cfg

    mock_run = MagicMock()
    mock_run.url = "https://console.example.com/run/123"
    mock_prefetch.return_value = mock_run

    result = runner.invoke(
        prefetch,
        ["hf-model", "meta-llama/Llama-2-7b-hf"],
        obj=mock_cfg,
    )

    assert result.exit_code == 0, result.output
    mock_prefetch.assert_called_once()
    call_kwargs = mock_prefetch.call_args.kwargs
    assert call_kwargs["repo"] == "meta-llama/Llama-2-7b-hf"
    assert call_kwargs["hf_token_key"] == "HF_TOKEN"  # default


@patch("flyte.prefetch.hf_model")
@patch("flyte.cli._run.initialize_config")
def test_hf_model_with_all_options(mock_init_config, mock_prefetch, runner, mock_cfg):
    """Test hf-model command with all options."""
    mock_init_config.return_value = mock_cfg

    mock_run = MagicMock()
    mock_run.url = "https://console.example.com/run/123"
    mock_prefetch.return_value = mock_run

    result = runner.invoke(
        prefetch,
        [
            "hf-model",
            "meta-llama/Llama-2-7b-hf",
            "--artifact-name",
            "llama-2-7b",
            "--architecture",
            "LlamaForCausalLM",
            "--task",
            "generate",
            "--modality",
            "text",
            "--modality",
            "image",
            "--format",
            "safetensors",
            "--model-type",
            "llama",
            "--short-description",
            "Test model",
            "--force",
            "1",
            "--hf-token-key",
            "MY_HF_TOKEN",
            "--cpu",
            "4",
            "--mem",
            "32Gi",
            "--ephemeral-storage",
            "100Gi",
            "--accelerator",
            "A100:1",
        ],
        obj=mock_cfg,
    )

    assert result.exit_code == 0, result.output
    mock_prefetch.assert_called_once()
    call_kwargs = mock_prefetch.call_args.kwargs
    assert call_kwargs["repo"] == "meta-llama/Llama-2-7b-hf"
    assert call_kwargs["artifact_name"] == "llama-2-7b"
    assert call_kwargs["architecture"] == "LlamaForCausalLM"
    assert call_kwargs["task"] == "generate"
    assert call_kwargs["modality"] == ("text", "image")
    assert call_kwargs["serial_format"] == "safetensors"
    assert call_kwargs["model_type"] == "llama"
    assert call_kwargs["short_description"] == "Test model"
    assert call_kwargs["force"] == 1
    assert call_kwargs["hf_token_key"] == "MY_HF_TOKEN"
    # Resources are now passed as a Resources object
    assert call_kwargs["resources"].cpu == "4"
    assert call_kwargs["resources"].memory == "32Gi"
    assert call_kwargs["resources"].disk == "100Gi"
    assert call_kwargs["resources"].gpu == "A100:1"


# =============================================================================
# Wait Flag Tests
# =============================================================================


@patch("flyte.prefetch.hf_model")
@patch("flyte.cli._run.initialize_config")
def test_hf_model_with_wait_flag(mock_init_config, mock_prefetch, runner, mock_cfg):
    """Test hf-model command with --wait flag."""
    mock_init_config.return_value = mock_cfg

    mock_run = MagicMock()
    mock_run.url = "https://console.example.com/run/123"
    mock_output = MagicMock()
    mock_output.path = "s3://bucket/model"
    mock_run.outputs.return_value = [mock_output]
    mock_prefetch.return_value = mock_run

    result = runner.invoke(
        prefetch,
        ["hf-model", "meta-llama/Llama-2-7b-hf", "--wait"],
        obj=mock_cfg,
    )

    assert result.exit_code == 0, result.output
    mock_run.wait.assert_called_once()
    assert "Model prefetched successfully" in result.output


@patch("flyte.prefetch.hf_model")
@patch("flyte.cli._run.initialize_config")
def test_hf_model_wait_with_failure(mock_init_config, mock_prefetch, runner, mock_cfg):
    """Test hf-model command with --wait when execution fails."""
    mock_init_config.return_value = mock_cfg

    mock_run = MagicMock()
    mock_run.url = "https://console.example.com/run/123"
    mock_run.outputs.side_effect = Exception("Run failed")
    mock_prefetch.return_value = mock_run

    result = runner.invoke(
        prefetch,
        ["hf-model", "meta-llama/Llama-2-7b-hf", "--wait"],
        obj=mock_cfg,
    )

    assert result.exit_code == 0, result.output  # CLI doesn't fail, just reports
    assert "Model prefetch failed" in result.output


# =============================================================================
# Multiple Modalities Tests
# =============================================================================


@patch("flyte.prefetch.hf_model")
@patch("flyte.cli._run.initialize_config")
def test_hf_model_single_modality(mock_init_config, mock_prefetch, runner, mock_cfg):
    """Test hf-model with single modality."""
    mock_init_config.return_value = mock_cfg

    mock_run = MagicMock()
    mock_run.url = "https://console.example.com/run/123"
    mock_prefetch.return_value = mock_run

    result = runner.invoke(
        prefetch,
        ["hf-model", "meta-llama/Llama-2-7b-hf", "--modality", "text"],
        obj=mock_cfg,
    )

    assert result.exit_code == 0, result.output
    call_kwargs = mock_prefetch.call_args.kwargs
    assert call_kwargs["modality"] == ("text",)


@patch("flyte.prefetch.hf_model")
@patch("flyte.cli._run.initialize_config")
def test_hf_model_multiple_modalities(mock_init_config, mock_prefetch, runner, mock_cfg):
    """Test hf-model with multiple modalities."""
    mock_init_config.return_value = mock_cfg

    mock_run = MagicMock()
    mock_run.url = "https://console.example.com/run/123"
    mock_prefetch.return_value = mock_run

    result = runner.invoke(
        prefetch,
        [
            "hf-model",
            "openai/clip-vit-base-patch32",
            "--modality",
            "text",
            "--modality",
            "image",
            "--modality",
            "audio",
        ],
        obj=mock_cfg,
    )

    assert result.exit_code == 0, result.output
    call_kwargs = mock_prefetch.call_args.kwargs
    assert call_kwargs["modality"] == ("text", "image", "audio")


@patch("flyte.prefetch.hf_model")
@patch("flyte.cli._run.initialize_config")
def test_hf_model_default_modality(mock_init_config, mock_prefetch, runner, mock_cfg):
    """Test hf-model uses default text modality when not specified."""
    mock_init_config.return_value = mock_cfg

    mock_run = MagicMock()
    mock_run.url = "https://console.example.com/run/123"
    mock_prefetch.return_value = mock_run

    result = runner.invoke(
        prefetch,
        ["hf-model", "meta-llama/Llama-2-7b-hf"],
        obj=mock_cfg,
    )

    assert result.exit_code == 0, result.output
    call_kwargs = mock_prefetch.call_args.kwargs
    assert call_kwargs["modality"] == ("text",)
