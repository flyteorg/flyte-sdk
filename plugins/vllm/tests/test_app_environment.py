"""Unit tests for VLLMAppEnvironment."""

import flyte
import flyte.app
import pytest
from flyte.app._parameter import Parameter
from flyte.models import SerializationContext

from flyteplugins.vllm import VLLMAppEnvironment
from flyteplugins.vllm._app_environment import DEFAULT_VLLM_IMAGE

# Tests for VLLMAppEnvironment initialization


def test_basic_init_with_model_path():
    """Test basic initialization with model_path."""
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
    )
    assert app.name == "test-app"
    assert app.model_path == "s3://bucket/model"
    assert app.model_id == "test-model"
    assert app.port.port == 8000
    assert app.type == "vLLM"
    assert app.stream_model is True
    assert app.image == DEFAULT_VLLM_IMAGE


def test_basic_init_with_model_hf_path():
    """Test basic initialization with model_hf_path."""
    app = VLLMAppEnvironment(
        name="test-app",
        model_hf_path="Qwen/Qwen3-0.6B",
        model_id="test-model",
    )
    assert app.name == "test-app"
    assert app.model_hf_path == "Qwen/Qwen3-0.6B"
    assert app.model_id == "test-model"
    assert app.port.port == 8000
    assert app.type == "vLLM"
    assert app.image == DEFAULT_VLLM_IMAGE
    # When using model_hf_path, no parameters should be created
    assert app.inputs == []
    # The model mount path should be set to the HF path
    assert app.env_vars["FLYTE_MODEL_LOADER_LOCAL_MODEL_PATH"] == "Qwen/Qwen3-0.6B"


def test_custom_image():
    """Test that custom image overrides the default."""
    custom_image = "my-registry/vllm:custom"
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        image=custom_image,
    )
    assert app.image == custom_image


def test_custom_port():
    """Test custom port configuration."""
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        port=8080,
    )
    assert app.port.port == 8080
    assert "--port" in app.args
    assert "8080" in app.args


# Tests for VLLMAppEnvironment validation


def test_missing_model_id_raises_error():
    """Test that missing model_id raises ValueError."""
    with pytest.raises(ValueError, match="model_id must be defined"):
        VLLMAppEnvironment(
            name="test-app",
            model_path="s3://bucket/model",
            model_id="",
        )


def test_missing_model_path_and_hf_path_raises_error():
    """Test that missing both model_path and model_hf_path raises ValueError."""
    with pytest.raises(ValueError, match="model_path or model_hf_path must be defined"):
        VLLMAppEnvironment(
            name="test-app",
            model_id="test-model",
        )


def test_both_model_path_and_hf_path_raises_error():
    """Test that setting both model_path and model_hf_path raises ValueError."""
    with pytest.raises(ValueError, match="model_path and model_hf_path cannot be set at the same time"):
        VLLMAppEnvironment(
            name="test-app",
            model_path="s3://bucket/model",
            model_hf_path="Qwen/Qwen3-0.6B",
            model_id="test-model",
        )


def test_args_set_raises_error():
    """Test that setting args raises ValueError."""
    with pytest.raises(ValueError, match="args cannot be set for VLLMAppEnvironment"):
        VLLMAppEnvironment(
            name="test-app",
            model_path="s3://bucket/model",
            model_id="test-model",
            args=["some", "args"],
        )


def test_inputs_set_raises_error():
    """Test that setting inputs raises ValueError."""
    with pytest.raises(ValueError, match="parameters cannot be set for VLLMAppEnvironment"):
        VLLMAppEnvironment(
            name="test-app",
            model_path="s3://bucket/model",
            model_id="test-model",
            parameters=[Parameter(name="foo", value="bar")],
        )


# Tests for stream_model configuration


def test_stream_model_true_with_model_path():
    """Test stream_model=True configuration with model_path."""
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        stream_model=True,
    )
    # Should have streaming load format
    assert "--load-format" in app.args
    assert "flyte-vllm-streaming" in app.args

    # Check env vars
    assert app.env_vars["FLYTE_MODEL_LOADER_STREAM_SAFETENSORS"] == "true"
    assert app.env_vars["FLYTE_MODEL_LOADER_LOCAL_MODEL_PATH"] == "/root/flyte"

    # Check parameters
    assert len(app.inputs) == 1
    model_input = app.inputs[0]
    assert model_input.name == "model"
    assert model_input.value == "s3://bucket/model"
    assert model_input.env_var == "FLYTE_MODEL_LOADER_REMOTE_MODEL_PATH"
    assert model_input.download is False


def test_stream_model_false_with_model_path():
    """Test stream_model=False configuration with model_path."""
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        stream_model=False,
    )
    # Should NOT have streaming load format
    assert "--load-format" not in app.args
    assert "flyte-vllm-streaming" not in app.args

    # Check env vars
    assert app.env_vars["FLYTE_MODEL_LOADER_STREAM_SAFETENSORS"] == "false"

    # Check parameters - should download instead of stream
    assert len(app.inputs) == 1
    model_input = app.inputs[0]
    assert model_input.download is True
    assert model_input.mount == "/root/flyte"


def test_model_hf_path_no_inputs():
    """Test that model_hf_path creates no parameters and sets correct mount path."""
    app = VLLMAppEnvironment(
        name="test-app",
        model_hf_path="meta-llama/Llama-2-7b",
        model_id="test-model",
    )

    # No parameters should be created for HF path
    assert app.inputs == []

    # Mount path should be set to the HF path
    assert app.env_vars["FLYTE_MODEL_LOADER_LOCAL_MODEL_PATH"] == "meta-llama/Llama-2-7b"


# Tests for extra_args configuration


def test_extra_args_as_string():
    """Test extra_args provided as a string."""
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        extra_args="--max-model-len 8192 --enforce-eager",
    )
    assert "--max-model-len" in app.args
    assert "8192" in app.args
    assert "--enforce-eager" in app.args


def test_extra_args_as_list():
    """Test extra_args provided as a list."""
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        extra_args=["--max-model-len", "4096", "--gpu-memory-utilization", "0.9"],
    )
    assert "--max-model-len" in app.args
    assert "4096" in app.args
    assert "--gpu-memory-utilization" in app.args
    assert "0.9" in app.args


def test_extra_args_empty_string():
    """Test extra_args as empty string (default)."""
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        extra_args="",
    )
    # Should have base args but no extra args
    assert "vllm-fserve" in app.args
    assert "serve" in app.args


# Tests for container_args method


def test_container_args_returns_list():
    """Test that container_args returns the args list."""
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
    )
    sctx = SerializationContext(version="123")
    result = app.container_args(sctx)

    assert isinstance(result, list)
    assert "vllm-fserve" in result
    assert "serve" in result
    assert "--served-model-name" in result
    assert "test-model" in result


def test_container_args_includes_port():
    """Test that container_args includes port."""
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        port=9000,
    )
    sctx = SerializationContext(version="123")
    result = app.container_args(sctx)

    port_idx = result.index("--port")
    assert result[port_idx + 1] == "9000"


# Tests for links configuration


def test_default_link_added():
    """Test that vLLM OpenAPI docs link is added by default."""
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
    )
    # First link should be the vLLM docs
    assert len(app.links) >= 1
    docs_link = app.links[0]
    assert docs_link.path == "/docs"
    assert docs_link.title == "vLLM OpenAPI Docs"
    assert docs_link.is_relative is True


def test_custom_links_preserved():
    """Test that custom links are preserved alongside default link."""
    custom_link = flyte.app.Link(path="/custom", title="Custom Link")
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        links=[custom_link],
    )
    # Should have default link first, then custom link
    assert len(app.links) == 2
    assert app.links[0].path == "/docs"
    assert app.links[1].path == "/custom"


# Tests for environment variables configuration


def test_env_vars_initialized_if_none():
    """Test that env_vars is initialized if None."""
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        env_vars=None,
    )
    assert app.env_vars is not None
    assert isinstance(app.env_vars, dict)


def test_custom_env_vars_preserved():
    """Test that custom env vars are preserved."""
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        env_vars={"MY_VAR": "my_value"},
    )
    assert app.env_vars["MY_VAR"] == "my_value"
    # Should also have the model loader env vars
    assert "FLYTE_MODEL_LOADER_LOCAL_MODEL_PATH" in app.env_vars
