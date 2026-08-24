"""Unit tests for SGLangAppEnvironment."""

import shlex

import flyte
import flyte.app
import pytest
from flyte.app._parameter import Parameter
from flyte.models import SerializationContext

from flyteplugins.sglang import SGLangAppEnvironment
from flyteplugins.sglang._app_environment import DEFAULT_SGLANG_IMAGE

# Tests for SGLangAppEnvironment initialization


def test_basic_init_with_model_path():
    """Test basic initialization with model_path."""
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
    )
    assert app.name == "test-app"
    assert app.model_path == "s3://bucket/model"
    assert app.model_id == "test-model"
    assert app.port.port == 8080
    assert app.type == "SGLang"
    assert app.stream_model is True
    assert app.image == DEFAULT_SGLANG_IMAGE


def test_basic_init_with_model_hf_path():
    """Test basic initialization with model_hf_path."""
    app = SGLangAppEnvironment(
        name="test-app",
        model_hf_path="Qwen/Qwen3-0.6B",
        model_id="test-model",
    )
    assert app.name == "test-app"
    assert app.model_hf_path == "Qwen/Qwen3-0.6B"
    assert app.model_id == "test-model"
    assert app.port.port == 8080
    assert app.type == "SGLang"
    assert app.stream_model is True
    assert app.image == DEFAULT_SGLANG_IMAGE
    # Hugging Face path: no Flyte blob streaming without model_path (matches sglang-fserve guards).
    assert app.env_vars["FLYTE_MODEL_LOADER_STREAM_SAFETENSORS"] == "false"
    # When using model_hf_path, no parameters should be created
    assert app.parameters == []
    # The model mount path should be set to the HF path
    assert app.env_vars["FLYTE_MODEL_LOADER_LOCAL_MODEL_PATH"] == "Qwen/Qwen3-0.6B"


def test_custom_image():
    """Test that custom image overrides the default."""
    custom_image = "my-registry/sglang:custom"
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        image=custom_image,
    )
    assert app.image == custom_image


def test_custom_port():
    """Test custom port configuration."""
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        port=8080,
    )
    assert app.port.port == 8080
    assert "--port" in app.args
    assert "8080" in app.args


# Tests for SGLangAppEnvironment validation


def test_missing_model_id_raises_error():
    """Test that missing model_id raises ValueError."""
    with pytest.raises(ValueError, match="model_id must be defined"):
        SGLangAppEnvironment(
            name="test-app",
            model_path="s3://bucket/model",
            model_id="",
        )


def test_missing_model_path_and_hf_path_raises_error():
    """Test that missing both model_path and model_hf_path raises ValueError."""
    with pytest.raises(ValueError, match="model_path or model_hf_path must be defined"):
        SGLangAppEnvironment(
            name="test-app",
            model_id="test-model",
        )


def test_both_model_path_and_hf_path_raises_error():
    """Test that setting both model_path and model_hf_path raises ValueError."""
    with pytest.raises(ValueError, match="model_path and model_hf_path cannot be set at the same time"):
        SGLangAppEnvironment(
            name="test-app",
            model_path="s3://bucket/model",
            model_hf_path="Qwen/Qwen3-0.6B",
            model_id="test-model",
        )


def test_args_set_raises_error():
    """Test that setting args raises ValueError."""
    with pytest.raises(ValueError, match="args cannot be set for SGLangAppEnvironment"):
        SGLangAppEnvironment(
            name="test-app",
            model_path="s3://bucket/model",
            model_id="test-model",
            args=["some", "args"],
        )


def test_inputs_set_raises_error():
    """Test that setting inputs raises ValueError."""
    with pytest.raises(ValueError, match="parameters cannot be set for SGLangAppEnvironment"):
        SGLangAppEnvironment(
            name="test-app",
            model_path="s3://bucket/model",
            model_id="test-model",
            parameters=[Parameter(name="foo", value="bar")],
        )


# Tests for stream_model configuration


def test_stream_model_true_with_model_path():
    """Test stream_model=True configuration with model_path."""
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        stream_model=True,
    )

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
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        stream_model=False,
    )

    # Check env vars
    assert app.env_vars["FLYTE_MODEL_LOADER_STREAM_SAFETENSORS"] == "false"

    # Check parameters - should download instead of stream
    assert len(app.parameters) == 1
    model_input = app.parameters[0]
    assert model_input.download is True
    assert model_input.mount == "/tmp/flyte/model"


def test_model_hf_path_no_inputs():
    """Test that model_hf_path creates no parameters and sets correct mount path."""
    app = SGLangAppEnvironment(
        name="test-app",
        model_hf_path="meta-llama/Llama-2-7b",
        model_id="test-model",
    )

    # No parameters should be created for HF path
    assert app.parameters == []

    # Streaming env off when there is no remote model_path
    assert app.env_vars["FLYTE_MODEL_LOADER_STREAM_SAFETENSORS"] == "false"

    # Mount path should be set to the HF path
    assert app.env_vars["FLYTE_MODEL_LOADER_LOCAL_MODEL_PATH"] == "meta-llama/Llama-2-7b"


# Tests for extra_args configuration


def test_extra_args_as_string():
    """Test extra_args provided as a string."""
    app = SGLangAppEnvironment(
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
    app = SGLangAppEnvironment(
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
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        extra_args="",
    )
    # Should have base args but no extra args
    assert "sglang-fserve" in app.args
    assert "--model-path" in app.args


# Tests for container_args method


def test_container_args_returns_list():
    """Test that container_args returns the args list."""
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
    )
    sctx = SerializationContext(version="123")
    result = app.container_args(sctx)

    assert isinstance(result, list)
    assert "sglang-fserve" in result
    assert "--model-path" in result
    assert "--served-model-name" in result
    assert "test-model" in result


def test_container_args_includes_port():
    """Test that container_args includes port."""
    app = SGLangAppEnvironment(
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
    """Test that SGLang OpenAPI docs link is added by default."""
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
    )
    # First link should be the SGLang docs
    assert len(app.links) >= 1
    docs_link = app.links[0]
    assert docs_link.path == "/docs"
    assert docs_link.title == "SGLang OpenAPI Docs"
    assert docs_link.is_relative is True


def test_custom_links_preserved():
    """Test that custom links are preserved alongside default link."""
    custom_link = flyte.app.Link(path="/custom", title="Custom Link")
    app = SGLangAppEnvironment(
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
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        env_vars=None,
    )
    assert app.env_vars is not None
    assert isinstance(app.env_vars, dict)


def test_custom_env_vars_preserved():
    """Test that custom env vars are preserved."""
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        env_vars={"MY_VAR": "my_value"},
    )
    assert app.env_vars["MY_VAR"] == "my_value"
    # Should also have the model loader env vars
    assert "FLYTE_MODEL_LOADER_LOCAL_MODEL_PATH" in app.env_vars


# Tests for server, on_startup, and on_shutdown validation


def _create_sglang_app_with_lifecycle_field(field_name, field_value):
    """Helper to create a SGLangAppEnvironment instance with a lifecycle field set before __post_init__."""
    app = object.__new__(SGLangAppEnvironment)
    app.name = "test-app"
    app.model_path = "s3://bucket/model"
    app.model_id = "test-model"
    app.port = 8080
    app.type = "SGLang"
    app.extra_args = ""
    app.stream_model = True
    app.image = DEFAULT_SGLANG_IMAGE
    app._model_mount_path = "/tmp/flyte/model"
    setattr(app, field_name, field_value)
    return app


def test_server_decorator_raises_error():
    """Test that setting _server raises ValueError in __post_init__."""
    app = _create_sglang_app_with_lifecycle_field("_server", lambda: None)
    with pytest.raises(ValueError, match="server function cannot be set for SGLangAppEnvironment"):
        SGLangAppEnvironment.__post_init__(app)


def test_on_startup_decorator_raises_error():
    """Test that setting _on_startup raises ValueError in __post_init__."""
    app = _create_sglang_app_with_lifecycle_field("_on_startup", lambda: None)
    with pytest.raises(ValueError, match="on_startup function cannot be set for SGLangAppEnvironment"):
        SGLangAppEnvironment.__post_init__(app)


def test_on_shutdown_decorator_raises_error():
    """Test that setting _on_shutdown raises ValueError in __post_init__."""
    app = _create_sglang_app_with_lifecycle_field("_on_shutdown", lambda: None)
    with pytest.raises(ValueError, match="on_shutdown function cannot be set for SGLangAppEnvironment"):
        SGLangAppEnvironment.__post_init__(app)


# Tests for host binding


def test_host_defaults_to_all_interfaces():
    """SGLang binds 127.0.0.1 by default, which is unreachable from outside the container."""
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
    )
    assert app.args[app.args.index("--host") + 1] == "0.0.0.0"


def test_explicit_host_is_not_overridden():
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        extra_args=["--host", "127.0.0.1"],
    )
    assert app.args.count("--host") == 1
    assert app.args[app.args.index("--host") + 1] == "127.0.0.1"


# Tests for speculative decoding / draft models


def test_draft_model_path_mounts_second_model():
    """A draft model is mounted alongside the target and passed as --speculative-draft-model-path."""
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        draft_model_path="s3://bucket/eagle3-head",
        speculative_config={"algorithm": "EAGLE3"},
    )

    assert len(app.parameters) == 2
    target, draft = app.parameters
    assert target.name == "model_path"
    assert target.mount == "/tmp/flyte/model"
    assert draft.name == "draft_model_path"
    assert draft.value == "s3://bucket/eagle3-head"
    assert draft.download is True
    assert draft.mount == "/tmp/flyte/draft-model"

    assert app.args[app.args.index("--speculative-algorithm") + 1] == "EAGLE3"
    assert app.args[app.args.index("--speculative-draft-model-path") + 1] == "/tmp/flyte/draft-model"


def test_speculative_config_renders_flat_flags():
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        draft_model_path="s3://bucket/dflash",
        speculative_config={"algorithm": "DFLASH", "num_draft_tokens": 16},
    )
    assert app.args[app.args.index("--speculative-num-draft-tokens") + 1] == "16"


def test_speculative_config_accepts_fully_spelled_keys():
    """Both "algorithm" and "speculative_algorithm" name the same flag."""
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        speculative_config={"speculative_algorithm": "NGRAM", "speculative_num_steps": 5},
    )
    assert app.args[app.args.index("--speculative-algorithm") + 1] == "NGRAM"
    assert app.args[app.args.index("--speculative-num-steps") + 1] == "5"


def test_speculative_config_boolean_flags():
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        speculative_config={"algorithm": "EAGLE3", "attention_mode": True, "disabled_thing": False},
    )
    assert "--speculative-attention-mode" in app.args
    assert "--speculative-disabled-thing" not in app.args


def test_draft_model_disables_streaming():
    """The Flyte loader monkeypatch is single-model, so a draft model forces download mode."""
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        stream_model=True,
        draft_model_path="s3://bucket/eagle3-head",
        speculative_config={"algorithm": "EAGLE3"},
    )
    assert app.env_vars["FLYTE_MODEL_LOADER_STREAM_SAFETENSORS"] == "false"
    assert app.parameters[0].download is True
    assert app.parameters[0].mount == "/tmp/flyte/model"


def test_draft_model_hf_path_creates_no_parameter():
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        draft_model_hf_path="Qwen/Qwen3-0.6B",
        speculative_config={"algorithm": "STANDALONE"},
    )
    assert [p.name for p in app.parameters] == ["model_path"]
    assert app.args[app.args.index("--speculative-draft-model-path") + 1] == "Qwen/Qwen3-0.6B"


def test_speculative_config_without_draft_model():
    """NGRAM speculation uses no draft model at all."""
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        speculative_config={"algorithm": "NGRAM", "num_draft_tokens": 8},
    )
    assert len(app.parameters) == 1
    assert "--speculative-draft-model-path" not in app.args
    # Streaming is unaffected: there is only one set of weights.
    assert app.env_vars["FLYTE_MODEL_LOADER_STREAM_SAFETENSORS"] == "true"


def test_draft_model_without_speculative_config_raises_error():
    with pytest.raises(ValueError, match="speculative_config must be defined when a draft model is set"):
        SGLangAppEnvironment(
            name="test-app",
            model_path="s3://bucket/model",
            model_id="test-model",
            draft_model_path="s3://bucket/eagle3-head",
        )


def test_speculative_config_draft_model_path_key_raises_error():
    with pytest.raises(ValueError, match="speculative_config cannot set 'draft_model_path'"):
        SGLangAppEnvironment(
            name="test-app",
            model_path="s3://bucket/model",
            model_id="test-model",
            speculative_config={"algorithm": "EAGLE3", "draft_model_path": "/somewhere/else"},
        )


def test_both_draft_model_path_and_hf_path_raises_error():
    with pytest.raises(ValueError, match="draft_model_path and draft_model_hf_path cannot be set at the same time"):
        SGLangAppEnvironment(
            name="test-app",
            model_path="s3://bucket/model",
            model_id="test-model",
            draft_model_path="s3://bucket/eagle3-head",
            draft_model_hf_path="Qwen/Qwen3-0.6B",
            speculative_config={"algorithm": "EAGLE3"},
        )


# Tests for the cache-aware router


def test_router_uses_router_entrypoint():
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        router=True,
        extra_args=["--dp-size", "4", "--router-policy", "cache_aware"],
    )
    assert app.args[:3] == ["python", "-m", "sglang_router.launch_server"]
    assert "--dp-size" in app.args
    assert "--router-policy" in app.args


def test_router_disables_streaming():
    """Router workers are separate processes that never load the patched loader."""
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        stream_model=True,
        router=True,
    )
    assert app.env_vars["FLYTE_MODEL_LOADER_STREAM_SAFETENSORS"] == "false"
    assert app.parameters[0].download is True


def test_default_entrypoint_is_the_fserve_shim():
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
    )
    assert app.args[0] == "sglang-fserve"


# Tests for shell-safe args


def test_extra_args_with_spaces_are_shell_quoted():
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        extra_args=["--json-model-override-args", '{"rope_scaling": {"factor": 2.0}}'],
    )
    assert shlex.split(" ".join(app.args))[-1] == '{"rope_scaling": {"factor": 2.0}}'


def test_env_var_args_are_left_unquoted():
    """fserve expands $VARS before joining; quoting would turn the marker into a literal."""
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        extra_args=["--api-key", "$SGLANG_API_KEY"],
    )
    assert "$SGLANG_API_KEY" in app.args


# Tests for clone_with with speculative decoding and routing


def test_clone_with_draft_model_and_router():
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
    )
    cloned = app.clone_with(
        name="spec-app",
        draft_model_path="s3://bucket/eagle3-head",
        speculative_config={"algorithm": "EAGLE3"},
        router=True,
    )
    assert cloned.router is True
    assert [p.name for p in cloned.parameters] == ["model_path", "draft_model_path"]
    # The original is untouched.
    assert app.router is False
    assert app.draft_model_path == ""


def test_clone_with_drops_draft_model():
    app = SGLangAppEnvironment(
        name="test-app",
        model_path="s3://bucket/model",
        model_id="test-model",
        draft_model_path="s3://bucket/eagle3-head",
        speculative_config={"algorithm": "EAGLE3"},
    )
    baseline = app.clone_with(name="baseline-app", draft_model_path=None, speculative_config=None)
    assert baseline.draft_model_path == ""
    assert [p.name for p in baseline.parameters] == ["model_path"]
    assert "--speculative-algorithm" not in baseline.args
