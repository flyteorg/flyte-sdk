"""Unit tests for RapidMLXAppEnvironment."""

import flyte
import flyte.app
import pytest
from flyte import Resources
from flyte.app._parameter import Parameter
from flyte.models import SerializationContext

from flyteplugins.rapidmlx import RapidMLXAppEnvironment
from flyteplugins.rapidmlx._app_environment import DEFAULT_RAPID_MLX_IMAGE
from flyteplugins.rapidmlx._constants import DEFAULT_RAPID_MLX_PORT

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_basic_init_minimal():
    """RapidMLXAppEnvironment can be constructed with just a name (with the default model)."""
    app = RapidMLXAppEnvironment(name="rapidmlx-app")
    assert app.name == "rapidmlx-app"
    assert app.port.port == DEFAULT_RAPID_MLX_PORT
    assert app.type == "Rapid-MLX"
    assert app.image == DEFAULT_RAPID_MLX_IMAGE
    assert app.model_id == "qwen3.5-4b-4bit"
    assert app.args[2] == app.model_id
    # No --model-dir or --api-key flags should be added.
    assert "--model-dir" not in app.args
    assert "--api-key" not in app.args
    # Core invocation should always be there.
    assert app.args[:2] == ["flyte-rapidmlx", "serve"]
    assert "--host" in app.args
    assert "--port" in app.args
    assert str(DEFAULT_RAPID_MLX_PORT) in app.args


def test_basic_init_with_model_id():
    app = RapidMLXAppEnvironment(name="rapidmlx-app", model_id="mlx-community/Qwen3.5-4B-MLX-4bit")
    assert app.args[2] == "mlx-community/Qwen3.5-4B-MLX-4bit"


def test_custom_port():
    app = RapidMLXAppEnvironment(name="rapidmlx-app", port=8123)
    assert app.port.port == 8123
    port_idx = app.args.index("--port")
    assert app.args[port_idx + 1] == "8123"


def test_custom_image():
    app = RapidMLXAppEnvironment(name="rapidmlx-app", image="my-registry/rapidmlx:custom")
    assert app.image == "my-registry/rapidmlx:custom"


def test_image_auto_resolves_to_default():
    app = RapidMLXAppEnvironment(name="rapidmlx-app", image="auto")
    assert app.image == DEFAULT_RAPID_MLX_IMAGE


# ---------------------------------------------------------------------------
# Apple Silicon integration (the devbox bridge)
# ---------------------------------------------------------------------------


def test_defaults_to_apple_silicon_gpu():
    """Rapid-MLX is mac-only, so no explicit gpu defaults to `gpu="AppleSilicon"`."""
    app = RapidMLXAppEnvironment(name="rapidmlx-app")
    assert app.resources is not None
    assert app.resources.gpu == "AppleSilicon"
    assert app.resources.get_device().device_class == "APPLE_SILICON"


def test_preserves_user_cpu_and_memory_and_adds_apple_silicon():
    """A user-supplied Resources with no gpu keeps its cpu/memory and gets AppleSilicon."""
    app = RapidMLXAppEnvironment(
        name="rapidmlx-app",
        resources=Resources(cpu=4, memory="16Gi"),
    )
    assert app.resources.cpu == 4
    assert app.resources.memory == "16Gi"
    assert app.resources.gpu == "AppleSilicon"


def test_explicit_apple_silicon_resources_kept():
    app = RapidMLXAppEnvironment(
        name="rapidmlx-app",
        resources=Resources(memory="32Gi", gpu="AppleSilicon"),
    )
    assert app.resources.gpu == "AppleSilicon"
    assert app.resources.memory == "32Gi"


def test_rejects_non_apple_silicon_gpu():
    """Rapid-MLX can't run on a CUDA GPU, so requesting one is an error."""
    with pytest.raises(ValueError, match="only runs on Apple Silicon"):
        RapidMLXAppEnvironment(name="rapidmlx-app", resources=Resources(gpu="T4:1"))


def test_rejects_explicit_gpu_count():
    with pytest.raises(ValueError, match="only runs on Apple Silicon"):
        RapidMLXAppEnvironment(name="rapidmlx-app", resources=Resources(gpu=1))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_args_set_raises_error():
    with pytest.raises(ValueError, match="args cannot be set for RapidMLXAppEnvironment"):
        RapidMLXAppEnvironment(name="rapidmlx-app", args=["something"])


def _build_with_lifecycle_field(field_name: str, field_value):
    app = object.__new__(RapidMLXAppEnvironment)
    app.name = "rapidmlx-app"
    app.port = DEFAULT_RAPID_MLX_PORT
    app.type = "Rapid-MLX"
    app.extra_args = ""
    app.model_id = ""
    app.api_key = ""
    app.image = DEFAULT_RAPID_MLX_IMAGE
    app._resolved_args = []
    setattr(app, field_name, field_value)
    return app


def test_server_func_raises_error():
    app = _build_with_lifecycle_field("_server", lambda: None)
    with pytest.raises(ValueError, match="server function cannot be set"):
        RapidMLXAppEnvironment.__post_init__(app)


def test_on_startup_raises_error():
    app = _build_with_lifecycle_field("_on_startup", lambda: None)
    with pytest.raises(ValueError, match="on_startup function cannot be set"):
        RapidMLXAppEnvironment.__post_init__(app)


def test_on_shutdown_raises_error():
    app = _build_with_lifecycle_field("_on_shutdown", lambda: None)
    with pytest.raises(ValueError, match="on_shutdown function cannot be set"):
        RapidMLXAppEnvironment.__post_init__(app)


# ---------------------------------------------------------------------------
# extra_args
# ---------------------------------------------------------------------------


def test_extra_args_string():
    app = RapidMLXAppEnvironment(
        name="rapidmlx-app",
        extra_args="--host 0.0.0.0",
    )
    assert "--host" in app.args
    assert "0.0.0.0" in app.args


@pytest.mark.parametrize("extra_args", ["--port 8123", "--port=8123", ["--port", "8123"], ["--port=8123"]])
def test_extra_args_cannot_override_port(extra_args):
    with pytest.raises(ValueError, match="Use `port`"):
        RapidMLXAppEnvironment(name="rapidmlx-app", extra_args=extra_args)


def test_clone_extra_args_cannot_override_port():
    app = RapidMLXAppEnvironment(name="rapidmlx-app")
    with pytest.raises(ValueError, match="Use `port`"):
        app.clone_with(name="rapidmlx-clone", extra_args="--port 8123")


def test_extra_args_list():
    app = RapidMLXAppEnvironment(
        name="rapidmlx-app",
        extra_args=["--host", "0.0.0.0"],
    )
    assert "--host" in app.args
    assert "0.0.0.0" in app.args


def test_extra_args_empty():
    app = RapidMLXAppEnvironment(name="rapidmlx-app", extra_args="")
    assert app.args[:2] == ["flyte-rapidmlx", "serve"]


def test_api_key_flag():
    app = RapidMLXAppEnvironment(name="rapidmlx-app", api_key="sk-test")
    api_idx = app.args.index("--api-key")
    assert app.args[api_idx + 1] == "sk-test"


def test_host_loopback_by_default():
    """Rapid-MLX is served on the Mac (devbox binds ports to loopback), so bind 127.0.0.1."""
    app = RapidMLXAppEnvironment(name="rapidmlx-app")
    host_idx = app.args.index("--host")
    assert app.args[host_idx + 1] == "127.0.0.1"


def test_host_override_in_extra_args():
    app = RapidMLXAppEnvironment(name="rapidmlx-app", extra_args="--host 0.0.0.0")
    assert app.args.count("--host") == 1
    assert app.args[app.args.index("--host") + 1] == "0.0.0.0"


def test_model_id_with_spaces_is_quoted_for_the_shell():
    """fserve runs the command through a shell, so paths with spaces survive quoting."""
    app = RapidMLXAppEnvironment(name="rapidmlx-app", model_id="/tmp/model files")
    assert app.args[2] == "'/tmp/model files'"


# ---------------------------------------------------------------------------
# container_args
# ---------------------------------------------------------------------------


def test_container_args_returns_list():
    app = RapidMLXAppEnvironment(name="rapidmlx-app", model_id="/tmp/models")
    sctx = SerializationContext(version="123")
    result = app.container_args(sctx)
    assert isinstance(result, list)
    assert result[:2] == ["flyte-rapidmlx", "serve"]
    assert result[2] == "/tmp/models"


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------


def test_default_link_added():
    app = RapidMLXAppEnvironment(name="rapidmlx-app")
    assert len(app.links) >= 1
    assert app.links[0].path == "/v1/models"
    assert app.links[0].title == "Rapid-MLX Models"
    assert app.links[0].is_relative is True


def test_custom_links_preserved():
    custom = flyte.app.Link(path="/custom", title="Custom")
    app = RapidMLXAppEnvironment(name="rapidmlx-app", links=[custom])
    assert len(app.links) == 2
    assert app.links[0].path == "/v1/models"
    assert app.links[1].path == "/custom"


def test_clone_preserves_links_without_duplicating_default():
    custom = flyte.app.Link(path="/custom", title="Custom")
    app = RapidMLXAppEnvironment(name="rapidmlx-app", links=[custom])
    original_links = list(app.links)
    cloned = app.clone_with(name="rapidmlx-clone")
    cloned_again = cloned.clone_with(name="rapidmlx-clone-again")
    assert app.links == cloned.links == cloned_again.links == original_links


def test_explicit_default_link_is_not_duplicated():
    link = flyte.app.Link(path="/v1/models", title="Rapid-MLX Models", is_relative=True)
    app = RapidMLXAppEnvironment(name="rapidmlx-app", links=[link])
    assert app.links == [link]


# ---------------------------------------------------------------------------
# env_vars
# ---------------------------------------------------------------------------


def test_env_vars_initialized_if_none():
    app = RapidMLXAppEnvironment(name="rapidmlx-app", env_vars=None)
    assert app.env_vars == {}


def test_custom_env_vars_preserved():
    app = RapidMLXAppEnvironment(name="rapidmlx-app", env_vars={"FOO": "bar"})
    assert app.env_vars["FOO"] == "bar"


# ---------------------------------------------------------------------------
# clone_with
# ---------------------------------------------------------------------------


def test_clone_with_overrides_model_id_and_port():
    app = RapidMLXAppEnvironment(name="rapidmlx-app", model_id="/a")
    cloned = app.clone_with(name="rapidmlx-app2", port=8200, model_id="/b")
    assert cloned.name == "rapidmlx-app2"
    assert cloned.model_id == "/b"
    assert cloned.port.port == 8200
    # The AppleSilicon default carries over via the (unmodified) resources.
    assert cloned.resources.gpu == "AppleSilicon"
    assert cloned.args[2] == "/b"
    port_idx = cloned.args.index("--port")
    assert cloned.args[port_idx + 1] == "8200"


def test_clone_with_unknown_kwarg_raises():
    app = RapidMLXAppEnvironment(name="rapidmlx-app")
    with pytest.raises(TypeError, match="Unexpected keyword arguments"):
        app.clone_with(name="rapidmlx-app2", what="ever")


def test_clone_with_carries_custom_resources():
    """A clone keeps a caller-supplied AppleSilicon Resources (cpu/memory intact)."""
    app = RapidMLXAppEnvironment(name="rapidmlx-app", resources=Resources(cpu=2, memory="8Gi"))
    cloned = app.clone_with(name="rapidmlx-app2")
    assert cloned.resources.cpu == 2
    assert cloned.resources.memory == "8Gi"
    assert cloned.resources.gpu == "AppleSilicon"


# ---------------------------------------------------------------------------
# parameters (we don't model parameters; ensure they aren't accidentally populated)
# ---------------------------------------------------------------------------


def test_parameters_default_empty():
    app = RapidMLXAppEnvironment(name="rapidmlx-app")
    assert app.parameters == []


def test_parameters_user_supplied_preserved():
    """Users may still attach parameters directly if they need to wire inputs."""
    p = Parameter(name="x", value="y")
    app = RapidMLXAppEnvironment(name="rapidmlx-app", parameters=[p])
    assert app.parameters == [p]


def test_clone_with_service_account():
    app = RapidMLXAppEnvironment(name="rapidmlx-app")
    cloned = app.clone_with(name="rapidmlx-app2", service_account="inference")
    assert cloned.service_account == "inference"
