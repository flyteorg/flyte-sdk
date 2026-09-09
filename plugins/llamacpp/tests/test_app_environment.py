"""Unit tests for LlamaCppAppEnvironment."""

import shlex

import flyte
import flyte.app
import pytest
from flyte.app._parameter import Parameter
from flyte.models import SerializationContext

from flyteplugins.llamacpp import LlamaCppAppEnvironment
from flyteplugins.llamacpp._image import DEFAULT_LLAMA_CPP_IMAGE

# Tests for LlamaCppAppEnvironment initialization


def test_basic_init_with_model_path():
    """Test basic initialization with model_path."""
    app = LlamaCppAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
    )
    assert app.name == "test-app"
    assert app.model_path == "s3://bucket/model"
    assert app.model_id == "test-model"
    assert app.port.port == 8080
    assert app.type == "llama.cpp"
    assert app.image == DEFAULT_LLAMA_CPP_IMAGE

    # Mounted weights are resolved by the shim at startup.
    assert "llama-cpp-fserve" in app.args
    assert "--model-dir" in app.args
    assert app.args[app.args.index("--model-dir") + 1] == "/tmp/flyte/model"
    assert "--alias" in app.args
    assert "test-model" in app.args

    # One downloaded/mounted parameter for the weights.
    assert len(app.parameters) == 1
    model_input = app.parameters[0]
    assert model_input.name == "model_path"
    assert model_input.value == "s3://bucket/model"
    assert model_input.download is True
    assert model_input.mount == "/tmp/flyte/model"


def test_basic_init_with_model_hf_path():
    """Test basic initialization with model_hf_path."""
    app = LlamaCppAppEnvironment(
        name="test-app",
        model_hf_path="ggml-org/gemma-3-4b-it-GGUF:Q4_K_M",
        model_id="gemma-3-4b-it",
    )
    assert app.model_hf_path == "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M"
    # HF repos are downloaded by llama-server itself; nothing is mounted.
    assert app.parameters == []
    assert "--hf-repo" in app.args
    assert app.args[app.args.index("--hf-repo") + 1] == "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M"
    assert "--model-dir" not in app.args


def test_custom_image():
    """Test that custom image overrides the default."""
    custom_image = "my-registry/llama-cpp:custom"
    app = LlamaCppAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        image=custom_image,
    )
    assert app.image == custom_image


def test_custom_port():
    """Test custom port configuration."""
    app = LlamaCppAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        port=9000,
    )
    assert app.port.port == 9000
    port_idx = app.args.index("--port")
    assert app.args[port_idx + 1] == "9000"


def test_host_binds_all_interfaces_by_default():
    """llama-server binds 127.0.0.1 by default, so the plugin adds --host 0.0.0.0."""
    app = LlamaCppAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
    )
    host_idx = app.args.index("--host")
    assert app.args[host_idx + 1] == "0.0.0.0"


def test_host_override_in_extra_args():
    """A --host in extra_args wins over the default."""
    app = LlamaCppAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        extra_args="--host 127.0.0.1",
    )
    assert app.args.count("--host") == 1
    assert app.args[app.args.index("--host") + 1] == "127.0.0.1"


# Tests for LlamaCppAppEnvironment validation


def test_missing_model_id_raises_error():
    """Test that missing model_id raises ValueError."""
    with pytest.raises(ValueError, match="model_id must be defined"):
        LlamaCppAppEnvironment(
            name="test-app",
            model_path="s3://bucket/model",
            model_id="",
        )


def test_missing_model_path_and_hf_path_raises_error():
    """Test that missing both model_path and model_hf_path raises ValueError."""
    with pytest.raises(ValueError, match="model_path or model_hf_path must be defined"):
        LlamaCppAppEnvironment(
            name="test-app",
            model_id="test-model",
        )


def test_both_model_path_and_hf_path_raises_error():
    """Test that setting both model_path and model_hf_path raises ValueError."""
    with pytest.raises(ValueError, match="model_path and model_hf_path cannot be set at the same time"):
        LlamaCppAppEnvironment(
            name="test-app",
            model_path="s3://bucket/model",
            model_hf_path="ggml-org/gemma-3-4b-it-GGUF",
            model_id="test-model",
        )


def test_args_set_raises_error():
    """Test that setting args raises ValueError."""
    with pytest.raises(ValueError, match="args cannot be set for LlamaCppAppEnvironment"):
        LlamaCppAppEnvironment(
            name="test-app",
            model_path="s3://bucket/model",
            model_id="test-model",
            args=["some", "args"],
        )


def test_inputs_set_raises_error():
    """Test that setting parameters raises ValueError."""
    with pytest.raises(ValueError, match="parameters cannot be set for LlamaCppAppEnvironment"):
        LlamaCppAppEnvironment(
            name="test-app",
            model_path="s3://bucket/model",
            model_id="test-model",
            parameters=[Parameter(name="foo", value="bar")],
        )


# Tests for extra_args configuration


def test_extra_args_as_string():
    """Test extra_args provided as a string."""
    app = LlamaCppAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        extra_args="--ctx-size 32768 --jinja",
    )
    assert "--ctx-size" in app.args
    assert "32768" in app.args
    assert "--jinja" in app.args


def test_extra_args_as_list():
    """Test extra_args provided as a list."""
    app = LlamaCppAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        extra_args=["--parallel", "4", "--flash-attn", "on"],
    )
    assert "--parallel" in app.args
    assert "4" in app.args
    assert "--flash-attn" in app.args
    assert "on" in app.args


# Tests for container_args method


def test_container_args_returns_list():
    """Test that container_args returns the args list."""
    app = LlamaCppAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
    )
    sctx = SerializationContext(version="123")
    result = app.container_args(sctx)

    assert isinstance(result, list)
    assert "llama-cpp-fserve" in result
    assert "--alias" in result
    assert "test-model" in result


# Tests for links configuration


def test_default_link_added():
    """Test that the llama.cpp Web UI link is added by default."""
    app = LlamaCppAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
    )
    assert len(app.links) >= 1
    ui_link = app.links[0]
    assert ui_link.path == "/"
    assert ui_link.title == "llama.cpp Web UI"
    assert ui_link.is_relative is True


def test_custom_links_preserved():
    """Test that custom links are preserved alongside default link."""
    custom_link = flyte.app.Link(path="/custom", title="Custom Link")
    app = LlamaCppAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        links=[custom_link],
    )
    assert len(app.links) == 2
    assert app.links[0].path == "/"
    assert app.links[1].path == "/custom"


# Tests for environment variables configuration


def test_env_vars_initialized_if_none():
    """Test that env_vars is initialized if None."""
    app = LlamaCppAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        env_vars=None,
    )
    assert app.env_vars is not None
    assert isinstance(app.env_vars, dict)


def test_custom_env_vars_preserved():
    """Test that custom env vars are preserved."""
    app = LlamaCppAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        env_vars={"MY_VAR": "my_value"},
    )
    assert app.env_vars["MY_VAR"] == "my_value"


# Tests for server, on_startup, and on_shutdown validation


def _create_app_with_lifecycle_field(field_name, field_value):
    """Helper to create a LlamaCppAppEnvironment with a lifecycle field set before __post_init__."""
    app = object.__new__(LlamaCppAppEnvironment)
    app.name = "test-app"
    app.model_path = "s3://bucket/model"
    app.model_id = "test-model"
    app.port = 8080
    app.type = "llama.cpp"
    app.extra_args = ""
    app.image = DEFAULT_LLAMA_CPP_IMAGE
    app._model_mount_path = "/tmp/flyte/model"
    app._draft_model_mount_path = "/tmp/flyte/draft-model"
    setattr(app, field_name, field_value)
    return app


def test_server_decorator_raises_error():
    """Test that setting _server raises ValueError in __post_init__."""
    app = _create_app_with_lifecycle_field("_server", lambda: None)
    with pytest.raises(ValueError, match="server function cannot be set for LlamaCppAppEnvironment"):
        LlamaCppAppEnvironment.__post_init__(app)


def test_on_startup_decorator_raises_error():
    """Test that setting _on_startup raises ValueError in __post_init__."""
    app = _create_app_with_lifecycle_field("_on_startup", lambda: None)
    with pytest.raises(ValueError, match="on_startup function cannot be set for LlamaCppAppEnvironment"):
        LlamaCppAppEnvironment.__post_init__(app)


def test_on_shutdown_decorator_raises_error():
    """Test that setting _on_shutdown raises ValueError in __post_init__."""
    app = _create_app_with_lifecycle_field("_on_shutdown", lambda: None)
    with pytest.raises(ValueError, match="on_shutdown function cannot be set for LlamaCppAppEnvironment"):
        LlamaCppAppEnvironment.__post_init__(app)


# Tests for speculative decoding / draft models


def test_draft_model_path_mounts_second_model():
    """A draft model is mounted alongside the target and resolved by the shim."""
    app = LlamaCppAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        draft_model_path="s3://bucket/draft-model",
    )

    assert len(app.parameters) == 2
    target, draft = app.parameters
    assert target.name == "model_path"
    assert target.mount == "/tmp/flyte/model"
    assert draft.name == "draft_model_path"
    assert draft.value == "s3://bucket/draft-model"
    assert draft.download is True
    assert draft.mount == "/tmp/flyte/draft-model"

    assert "--draft-model-dir" in app.args
    assert app.args[app.args.index("--draft-model-dir") + 1] == "/tmp/flyte/draft-model"


def test_draft_model_hf_path_creates_no_parameter():
    """A Hugging Face draft model is resolved by llama-server itself; nothing is mounted."""
    app = LlamaCppAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        draft_model_hf_path="ggml-org/Qwen3-0.6B-GGUF:Q8_0",
    )
    assert [p.name for p in app.parameters] == ["model_path"]
    assert "--hf-repo-draft" in app.args
    assert app.args[app.args.index("--hf-repo-draft") + 1] == "ggml-org/Qwen3-0.6B-GGUF:Q8_0"
    assert "--draft-model-dir" not in app.args


def test_both_draft_model_path_and_hf_path_raises_error():
    with pytest.raises(ValueError, match="draft_model_path and draft_model_hf_path cannot be set at the same time"):
        LlamaCppAppEnvironment(
            name="test-app",
            model_path="s3://bucket/model",
            model_id="test-model",
            draft_model_path="s3://bucket/draft-model",
            draft_model_hf_path="ggml-org/Qwen3-0.6B-GGUF:Q8_0",
        )


# Tests for shell-safe args


def test_extra_args_with_spaces_are_shell_quoted():
    app = LlamaCppAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        extra_args=["--chat-template-kwargs", '{"enable_thinking": false}'],
    )
    assert shlex.split(" ".join(app.args))[-1] == '{"enable_thinking": false}'


def test_ordinary_args_are_not_quoted():
    """shlex.quote is the identity for ordinary tokens, so plain args stay readable."""
    app = LlamaCppAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        extra_args="--ctx-size 8192",
    )
    assert "--ctx-size" in app.args
    assert "8192" in app.args
    assert "llama-cpp-fserve" in app.args


def test_env_var_args_are_left_unquoted():
    """fserve expands $VARS before joining; quoting would turn the marker into a literal."""
    app = LlamaCppAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        extra_args=["--api-key", "$LLAMA_API_KEY"],
    )
    assert "$LLAMA_API_KEY" in app.args


# Tests for clone_with


def test_clone_with_draft_model():
    app = LlamaCppAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
    )
    cloned = app.clone_with(
        name="spec-app",
        draft_model_path="s3://bucket/draft-model",
        extra_args="--draft-max 16",
    )
    assert cloned.name == "spec-app"
    assert cloned.draft_model_path == "s3://bucket/draft-model"
    assert [p.name for p in cloned.parameters] == ["model_path", "draft_model_path"]
    assert "--draft-max" in cloned.args
    # The original is untouched.
    assert app.draft_model_path == ""
    assert [p.name for p in app.parameters] == ["model_path"]


def test_clone_with_drops_draft_model():
    app = LlamaCppAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        draft_model_path="s3://bucket/draft-model",
    )
    baseline = app.clone_with(name="baseline-app", draft_model_path=None)
    assert baseline.draft_model_path == ""
    assert [p.name for p in baseline.parameters] == ["model_path"]
    assert "--draft-model-dir" not in baseline.args


def test_clone_with_switches_to_hf_path():
    app = LlamaCppAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
    )
    cloned = app.clone_with(
        name="hf-app",
        model_path=None,
        model_hf_path="ggml-org/gemma-3-4b-it-GGUF:Q4_K_M",
    )
    assert cloned.model_path == ""
    assert cloned.model_hf_path == "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M"
    assert not cloned.parameters
    assert "--hf-repo" in cloned.args
