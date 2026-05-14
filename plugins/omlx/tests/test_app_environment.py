"""Unit tests for OMLXAppEnvironment."""

import flyte
import flyte.app
import pytest
from flyte.app._parameter import Parameter
from flyte.models import SerializationContext

from flyteplugins.omlx import OMLXAppEnvironment
from flyteplugins.omlx._app_environment import DEFAULT_OMLX_IMAGE
from flyteplugins.omlx._constants import DEFAULT_OMLX_PORT


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_basic_init_minimal():
    """OMLXAppEnvironment can be constructed with just a name (no model_dir)."""
    app = OMLXAppEnvironment(name="omlx-app")
    assert app.name == "omlx-app"
    assert app.port.port == DEFAULT_OMLX_PORT
    assert app.type == "oMLX"
    assert app.image == DEFAULT_OMLX_IMAGE
    assert app.model_dir == ""
    assert app.model_id == ""
    # No --model-dir or --api-key flags should be added.
    assert "--model-dir" not in app.args
    assert "--api-key" not in app.args
    # Core invocation should always be there.
    assert app.args[:2] == ["flyte-omlx", "serve"]
    assert "--host" in app.args
    assert "--port" in app.args
    assert str(DEFAULT_OMLX_PORT) in app.args


def test_basic_init_with_model_dir():
    app = OMLXAppEnvironment(
        name="omlx-app",
        model_dir="/Users/me/.omlx/models",
        model_id="qwen3-0.6b",
    )
    assert app.model_dir == "/Users/me/.omlx/models"
    assert app.model_id == "qwen3-0.6b"
    md_idx = app.args.index("--model-dir")
    assert app.args[md_idx + 1] == "/Users/me/.omlx/models"


def test_custom_port():
    app = OMLXAppEnvironment(name="omlx-app", port=8123)
    assert app.port.port == 8123
    port_idx = app.args.index("--port")
    assert app.args[port_idx + 1] == "8123"


def test_custom_image():
    app = OMLXAppEnvironment(name="omlx-app", image="my-registry/omlx:custom")
    assert app.image == "my-registry/omlx:custom"


def test_image_auto_resolves_to_default():
    app = OMLXAppEnvironment(name="omlx-app", image="auto")
    assert app.image == DEFAULT_OMLX_IMAGE


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_args_set_raises_error():
    with pytest.raises(ValueError, match="args cannot be set for OMLXAppEnvironment"):
        OMLXAppEnvironment(name="omlx-app", args=["something"])


def _build_with_lifecycle_field(field_name: str, field_value):
    app = object.__new__(OMLXAppEnvironment)
    app.name = "omlx-app"
    app.port = DEFAULT_OMLX_PORT
    app.type = "oMLX"
    app.extra_args = ""
    app.model_dir = ""
    app.model_id = ""
    app.api_key = ""
    app.image = DEFAULT_OMLX_IMAGE
    app._resolved_args = []
    setattr(app, field_name, field_value)
    return app


def test_server_func_raises_error():
    app = _build_with_lifecycle_field("_server", lambda: None)
    with pytest.raises(ValueError, match="server function cannot be set"):
        OMLXAppEnvironment.__post_init__(app)


def test_on_startup_raises_error():
    app = _build_with_lifecycle_field("_on_startup", lambda: None)
    with pytest.raises(ValueError, match="on_startup function cannot be set"):
        OMLXAppEnvironment.__post_init__(app)


def test_on_shutdown_raises_error():
    app = _build_with_lifecycle_field("_on_shutdown", lambda: None)
    with pytest.raises(ValueError, match="on_shutdown function cannot be set"):
        OMLXAppEnvironment.__post_init__(app)


# ---------------------------------------------------------------------------
# extra_args
# ---------------------------------------------------------------------------


def test_extra_args_string():
    app = OMLXAppEnvironment(
        name="omlx-app",
        extra_args="--max-model-memory 32GB --log-level debug",
    )
    assert "--max-model-memory" in app.args
    assert "32GB" in app.args
    assert "--log-level" in app.args
    assert "debug" in app.args


def test_extra_args_list():
    app = OMLXAppEnvironment(
        name="omlx-app",
        extra_args=["--max-concurrent-requests", "16"],
    )
    assert "--max-concurrent-requests" in app.args
    assert "16" in app.args


def test_extra_args_empty():
    app = OMLXAppEnvironment(name="omlx-app", extra_args="")
    assert app.args[:2] == ["flyte-omlx", "serve"]


def test_api_key_flag():
    app = OMLXAppEnvironment(name="omlx-app", api_key="sk-test")
    api_idx = app.args.index("--api-key")
    assert app.args[api_idx + 1] == "sk-test"


# ---------------------------------------------------------------------------
# container_args
# ---------------------------------------------------------------------------


def test_container_args_returns_list():
    app = OMLXAppEnvironment(name="omlx-app", model_dir="/tmp/models")
    sctx = SerializationContext(version="123")
    result = app.container_args(sctx)
    assert isinstance(result, list)
    assert result[:2] == ["flyte-omlx", "serve"]
    assert "--model-dir" in result
    assert "/tmp/models" in result


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------


def test_default_link_added():
    app = OMLXAppEnvironment(name="omlx-app")
    assert len(app.links) >= 1
    assert app.links[0].path == "/v1/models"
    assert app.links[0].title == "oMLX Models"
    assert app.links[0].is_relative is True


def test_custom_links_preserved():
    custom = flyte.app.Link(path="/custom", title="Custom")
    app = OMLXAppEnvironment(name="omlx-app", links=[custom])
    assert len(app.links) == 2
    assert app.links[0].path == "/v1/models"
    assert app.links[1].path == "/custom"


# ---------------------------------------------------------------------------
# env_vars
# ---------------------------------------------------------------------------


def test_env_vars_initialized_if_none():
    app = OMLXAppEnvironment(name="omlx-app", env_vars=None)
    assert app.env_vars == {}


def test_custom_env_vars_preserved():
    app = OMLXAppEnvironment(name="omlx-app", env_vars={"FOO": "bar"})
    assert app.env_vars["FOO"] == "bar"


# ---------------------------------------------------------------------------
# clone_with
# ---------------------------------------------------------------------------


def test_clone_with_overrides_model_dir_and_port():
    app = OMLXAppEnvironment(name="omlx-app", model_dir="/a")
    cloned = app.clone_with(name="omlx-app2", port=8200, model_dir="/b")
    assert cloned.name == "omlx-app2"
    assert cloned.model_dir == "/b"
    assert cloned.port.port == 8200
    md_idx = cloned.args.index("--model-dir")
    assert cloned.args[md_idx + 1] == "/b"
    port_idx = cloned.args.index("--port")
    assert cloned.args[port_idx + 1] == "8200"


def test_clone_with_unknown_kwarg_raises():
    app = OMLXAppEnvironment(name="omlx-app")
    with pytest.raises(TypeError, match="Unexpected keyword arguments"):
        app.clone_with(name="omlx-app2", what="ever")


# ---------------------------------------------------------------------------
# parameters (we don't model parameters; just ensure they don't get accidentally
# populated by __post_init__)
# ---------------------------------------------------------------------------


def test_parameters_default_empty():
    app = OMLXAppEnvironment(name="omlx-app")
    assert app.parameters == []


def test_parameters_user_supplied_preserved():
    """Users may still attach parameters directly if they need to wire inputs."""
    p = Parameter(name="x", value="y")
    app = OMLXAppEnvironment(name="omlx-app", parameters=[p])
    assert app.parameters == [p]
