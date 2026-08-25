"""Unit tests for VLLMAppEnvironment."""

import json
import shlex

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
    assert app.port.port == 8080
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
    assert app.port.port == 8080
    assert app.type == "vLLM"
    assert app.stream_model is True
    assert app.image == DEFAULT_VLLM_IMAGE
    # Hugging Face path uses vLLM's default loading (no Flyte blob streaming without model_path).
    assert "--load-format" not in app.args
    assert app.env_vars["FLYTE_MODEL_LOADER_STREAM_SAFETENSORS"] == "false"
    # When using model_hf_path, no parameters should be created
    assert app.parameters == []
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
    assert app.env_vars["FLYTE_MODEL_LOADER_LOCAL_MODEL_PATH"] == "/tmp/flyte/model"

    # Check parameters
    assert len(app.parameters) == 1
    model_input = app.parameters[0]
    assert model_input.name == "model_path"
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
    assert len(app.parameters) == 1
    model_input = app.parameters[0]
    assert model_input.download is True
    assert model_input.mount == "/tmp/flyte/model"


def test_model_hf_path_no_inputs():
    """Test that model_hf_path creates no parameters and sets correct mount path."""
    app = VLLMAppEnvironment(
        name="test-app",
        model_hf_path="meta-llama/Llama-2-7b",
        model_id="test-model",
    )

    # No parameters should be created for HF path
    assert app.parameters == []

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


# Tests for server, on_startup, and on_shutdown validation


def _create_vllm_app_with_lifecycle_field(field_name, field_value):
    """Helper to create a VLLMAppEnvironment instance with a lifecycle field set before __post_init__."""
    app = object.__new__(VLLMAppEnvironment)
    app.name = "test-app"
    app.model_path = "s3://bucket/model"
    app.model_id = "test-model"
    app.port = 8080
    app.type = "vLLM"
    app.extra_args = ""
    app.stream_model = True
    app.image = DEFAULT_VLLM_IMAGE
    app._model_mount_path = "/tmp/flyte/model"
    setattr(app, field_name, field_value)
    return app


def test_server_decorator_raises_error():
    """Test that setting _server raises ValueError in __post_init__."""
    app = _create_vllm_app_with_lifecycle_field("_server", lambda: None)
    with pytest.raises(ValueError, match="server function cannot be set for VLLMAppEnvironment"):
        VLLMAppEnvironment.__post_init__(app)


def test_on_startup_decorator_raises_error():
    """Test that setting _on_startup raises ValueError in __post_init__."""
    app = _create_vllm_app_with_lifecycle_field("_on_startup", lambda: None)
    with pytest.raises(ValueError, match="on_startup function cannot be set for VLLMAppEnvironment"):
        VLLMAppEnvironment.__post_init__(app)


def test_on_shutdown_decorator_raises_error():
    """Test that setting _on_shutdown raises ValueError in __post_init__."""
    app = _create_vllm_app_with_lifecycle_field("_on_shutdown", lambda: None)
    with pytest.raises(ValueError, match="on_shutdown function cannot be set for VLLMAppEnvironment"):
        VLLMAppEnvironment.__post_init__(app)


# Tests for speculative decoding / draft models


def test_draft_model_path_mounts_second_model():
    """A draft model is mounted alongside the target and referenced by --speculative-config."""
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        draft_model_path="s3://bucket/eagle3-head",
        speculative_config={"method": "eagle3", "num_speculative_tokens": 3},
    )

    assert len(app.parameters) == 2
    target, draft = app.parameters
    assert target.name == "model_path"
    assert target.mount == "/tmp/flyte/model"
    assert draft.name == "draft_model_path"
    assert draft.value == "s3://bucket/eagle3-head"
    assert draft.download is True
    assert draft.mount == "/tmp/flyte/draft-model"

    # The draft model is passed to vLLM as the `model` key of the speculative config.
    config = json.loads(shlex.split(" ".join(app.args))[app.args.index("--speculative-config") + 1])
    assert config == {"method": "eagle3", "num_speculative_tokens": 3, "model": "/tmp/flyte/draft-model"}


def test_draft_model_disables_streaming():
    """The Flyte streaming loader is single-model, so a draft model forces download mode."""
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        stream_model=True,
        draft_model_path="s3://bucket/eagle3-head",
        speculative_config={"method": "eagle3", "num_speculative_tokens": 3},
    )
    assert "--load-format" not in app.args
    assert "flyte-vllm-streaming" not in app.args
    assert app.env_vars["FLYTE_MODEL_LOADER_STREAM_SAFETENSORS"] == "false"
    assert app.parameters[0].download is True
    assert app.parameters[0].mount == "/tmp/flyte/model"


def test_draft_model_hf_path_creates_no_parameter():
    """A Hugging Face draft model is resolved by vLLM itself, so nothing is mounted for it."""
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        draft_model_hf_path="Qwen/Qwen3-0.6B",
        speculative_config={"num_speculative_tokens": 5},
    )
    assert [p.name for p in app.parameters] == ["model_path"]
    config = json.loads(shlex.split(" ".join(app.args))[app.args.index("--speculative-config") + 1])
    assert config == {"num_speculative_tokens": 5, "model": "Qwen/Qwen3-0.6B"}


def test_speculative_config_without_draft_model():
    """Draft-model-free methods (ngram) need no mounted weights."""
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        speculative_config={"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4},
    )
    assert len(app.parameters) == 1
    config = json.loads(shlex.split(" ".join(app.args))[app.args.index("--speculative-config") + 1])
    assert config == {"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}
    # Streaming is unaffected: there is only one set of weights.
    assert "flyte-vllm-streaming" in app.args


def test_draft_model_without_speculative_config_raises_error():
    with pytest.raises(ValueError, match="speculative_config must be defined when a draft model is set"):
        VLLMAppEnvironment(
            name="test-app",
            model_path="s3://bucket/model",
            model_id="test-model",
            draft_model_path="s3://bucket/eagle3-head",
        )


def test_speculative_config_model_key_raises_error():
    with pytest.raises(ValueError, match="speculative_config cannot set 'model'"):
        VLLMAppEnvironment(
            name="test-app",
            model_path="s3://bucket/model",
            model_id="test-model",
            draft_model_path="s3://bucket/eagle3-head",
            speculative_config={"method": "eagle3", "model": "/somewhere/else"},
        )


def test_both_draft_model_path_and_hf_path_raises_error():
    with pytest.raises(ValueError, match="draft_model_path and draft_model_hf_path cannot be set at the same time"):
        VLLMAppEnvironment(
            name="test-app",
            model_path="s3://bucket/model",
            model_id="test-model",
            draft_model_path="s3://bucket/eagle3-head",
            draft_model_hf_path="Qwen/Qwen3-0.6B",
            speculative_config={"method": "eagle3"},
        )


# Tests for shell-safe args


def test_speculative_config_json_is_shell_quoted():
    """fserve joins args and runs them through a shell, so the JSON blob must be quoted."""
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        draft_model_path="s3://bucket/eagle3-head",
        speculative_config={"method": "eagle3", "num_speculative_tokens": 3},
    )
    blob = app.args[app.args.index("--speculative-config") + 1]
    assert blob.startswith("'") and blob.endswith("'")
    # What the shell hands to vLLM round-trips back to the original config.
    assert json.loads(shlex.split(blob)[0])["method"] == "eagle3"


def test_extra_args_with_spaces_are_shell_quoted():
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        extra_args=["--chat-template", '{"foo": "bar"}'],
    )
    assert shlex.split(" ".join(app.args))[-1] == '{"foo": "bar"}'


def test_ordinary_args_are_not_quoted():
    """shlex.quote is the identity for ordinary tokens, so plain args stay readable."""
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        extra_args="--max-model-len 8192",
    )
    assert "--max-model-len" in app.args
    assert "8192" in app.args
    assert "vllm-fserve" in app.args


def test_env_var_args_are_left_unquoted():
    """fserve expands $VARS before joining; quoting would turn the marker into a literal."""
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        extra_args=["--api-key", "$VLLM_API_KEY"],
    )
    assert "$VLLM_API_KEY" in app.args


# Tests for clone_with with speculative decoding


def test_clone_with_draft_model():
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
    )
    cloned = app.clone_with(
        name="spec-app",
        draft_model_path="s3://bucket/eagle3-head",
        speculative_config={"method": "eagle3", "num_speculative_tokens": 3},
    )
    assert cloned.name == "spec-app"
    assert cloned.draft_model_path == "s3://bucket/eagle3-head"
    assert [p.name for p in cloned.parameters] == ["model_path", "draft_model_path"]
    # The original is untouched.
    assert app.draft_model_path == ""
    assert [p.name for p in app.parameters] == ["model_path"]


def test_clone_with_drops_draft_model():
    app = VLLMAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        draft_model_path="s3://bucket/eagle3-head",
        speculative_config={"method": "eagle3", "num_speculative_tokens": 3},
    )
    baseline = app.clone_with(name="baseline-app", draft_model_path=None, speculative_config=None)
    assert baseline.draft_model_path == ""
    assert baseline.speculative_config is None
    assert [p.name for p in baseline.parameters] == ["model_path"]
    assert "--speculative-config" not in baseline.args
