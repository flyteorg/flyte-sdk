"""
Unit tests for flyte._serve module.

These tests verify the serve context functionality including:
- _Serve class initialization and configuration
- with_servecontext() function
- Parameter value override handling
"""

from dataclasses import replace
from unittest.mock import patch

import pytest

from flyte._image import Image
from flyte._serve import _Serve, with_servecontext
from flyte.app import AppEnvironment
from flyte.app._parameter import Parameter


def test_serve_default_initialization():
    """
    GOAL: Verify _Serve initializes with correct default values.

    Tests that all default parameters are set correctly when not explicitly provided.
    """
    serve = _Serve()
    assert serve._version is None
    assert serve._copy_style == "loaded_modules"
    assert serve._dry_run is False
    assert serve._project is None
    assert serve._domain is None
    assert serve._env_vars == {}
    assert serve._input_values == {}
    assert serve._cluster_pool is None
    assert serve._log_level is None
    assert serve._log_format == "console"
    assert serve._interactive_mode is False


def test_serve_with_custom_parameters():
    """
    GOAL: Verify _Serve accepts and stores all custom parameters.

    Tests that all parameters passed to _Serve are correctly stored.
    """

    serve = _Serve(
        version="v1.0.0",
        copy_style="all",
        dry_run=True,
        project="my-project",
        domain="production",
        env_vars={"API_KEY": "secret"},
        input_values={"my-app": {"config": "new-config.yaml"}},
        cluster_pool="gpu-pool",
        log_level=10,  # logging.DEBUG
        log_format="json",
    )

    assert serve._version == "v1.0.0"
    assert serve._copy_style == "all"
    assert serve._dry_run is True
    assert serve._project == "my-project"
    assert serve._domain == "production"
    assert serve._env_vars == {"API_KEY": "secret"}
    assert serve._input_values == {"my-app": {"config": "new-config.yaml"}}
    assert serve._cluster_pool == "gpu-pool"
    assert serve._log_level == 10
    assert serve._log_format == "json"


def test_serve_raises_in_interactive_mode():
    """
    GOAL: Verify _Serve raises NotImplementedError in interactive mode.

    Tests that apps cannot be served from notebooks or interactive environments.
    """
    with patch("flyte._serve.ipython_check", return_value=True):
        with pytest.raises(NotImplementedError, match="Apps do not support running from notebooks"):
            _Serve()


def test_serve_explicit_interactive_mode_true():
    """
    GOAL: Verify _Serve raises when interactive_mode is explicitly True.

    Tests that even with ipython_check returning False, explicit
    interactive_mode=True raises NotImplementedError.
    """
    with patch("flyte._serve.ipython_check", return_value=False):
        with pytest.raises(NotImplementedError, match="Apps do not support running from notebooks"):
            _Serve(interactive_mode=True)


def test_serve_explicit_interactive_mode_false():
    """
    GOAL: Verify _Serve works when interactive_mode is explicitly False.

    Tests that explicit interactive_mode=False bypasses the ipython_check.
    """
    with patch("flyte._serve.ipython_check", return_value=True):
        # This should NOT raise because interactive_mode is explicitly False
        serve = _Serve(interactive_mode=False)
        assert not serve._interactive_mode


# =============================================================================
# Tests for with_servecontext function
# =============================================================================


def test_with_servecontext_passes_all_parameters():
    """
    GOAL: Verify with_servecontext passes all parameters to _Serve.

    Tests that all parameters are correctly passed through to the _Serve constructor.
    """
    serve = with_servecontext(
        version="v2.0.0",
        copy_style="none",
        dry_run=True,
        project="test-project",
        domain="staging",
        env_vars={"ENV": "staging"},
        input_values={"app": {"key": "value"}},
        cluster_pool="cpu-pool",
        log_level=20,  # logging.INFO
        log_format="json",
    )

    assert serve._version == "v2.0.0"
    assert serve._copy_style == "none"
    assert serve._dry_run is True
    assert serve._project == "test-project"
    assert serve._domain == "staging"
    assert serve._env_vars == {"ENV": "staging"}
    assert serve._input_values == {"app": {"key": "value"}}
    assert serve._cluster_pool == "cpu-pool"
    assert serve._log_level == 20
    assert serve._log_format == "json"


# =============================================================================
# Tests for input value override functionality in _Serve
# =============================================================================


def test_input_values_dict_structure():
    """
    GOAL: Verify input_values dict has correct structure.

    Tests that input_values maps app environment names to dicts of input names to values.
    """
    import flyte.io

    input_values = {
        "app-one": {
            "config": "config-one.yaml",
            "data": flyte.io.Dir(path="s3://bucket/data-one"),
        },
        "app-two": {
            "model": flyte.io.File(path="s3://bucket/model.pkl"),
        },
    }

    serve = _Serve(input_values=input_values)

    assert "app-one" in serve._input_values
    assert "app-two" in serve._input_values
    assert serve._input_values["app-one"]["config"] == "config-one.yaml"
    assert isinstance(serve._input_values["app-one"]["data"], flyte.io.Dir)
    assert isinstance(serve._input_values["app-two"]["model"], flyte.io.File)


@pytest.mark.asyncio
async def test_serve_extracts_parameter_overrides_for_matching_app():
    """
    GOAL: Verify serve method correctly extracts input overrides for matching app.

    Tests that when input_values contains entries for the app being served,
    the overrides are correctly extracted and applied.
    """

    # Create an app environment with inputs
    app_env = AppEnvironment(
        name="my-test-app",
        image=Image.from_base("python:3.11"),
            parameters=[
            Parameter(value="original-config.yaml", name="config"),
            Parameter(value="s3://original/data", name="data"),
        ],
    )

    # Create serve context with input overrides for this app
    input_values = {
        "my-test-app": {
            "config": "overridden-config.yaml",
            "data": "s3://new/data",
        }
    }

    serve = _Serve(input_values=input_values)

    # Manually test the override extraction logic (from _Serve.serve method)
    app_env_input_values = serve._input_values.get(app_env.name)
    assert app_env_input_values is not None

    parameter_overrides = []
    for _input in app_env.inputs:
        value = app_env_input_values.get(_input.name, _input.value)
        parameter_overrides.append(replace(_input, value=value))

    # Verify overrides were created correctly
    assert len(parameter_overrides) == 2
    assert parameter_overrides[0].name == "config"
    assert parameter_overrides[0].value == "overridden-config.yaml"
    assert parameter_overrides[1].name == "data"
    assert parameter_overrides[1].value == "s3://new/data"


@pytest.mark.asyncio
async def test_serve_no_overrides_for_non_matching_app():
    """
    GOAL: Verify serve method returns None for apps not in input_values.

    Tests that when input_values doesn't contain entries for the app being served,
    no overrides are extracted.
    """
    # Create an app environment
    app_env = AppEnvironment(
        name="different-app",
        image=Image.from_base("python:3.11"),
            parameters=[
            Parameter(value="config.yaml", name="config"),
        ],
    )

    # Create serve context with input overrides for a DIFFERENT app
    input_values = {
        "other-app": {
            "config": "other-config.yaml",
        }
    }

    serve = _Serve(input_values=input_values)

    # Test the override extraction logic
    app_env_input_values = serve._input_values.get(app_env.name)
    assert app_env_input_values is None  # No overrides for this app


@pytest.mark.asyncio
async def test_serve_partial_parameter_overrides():
    """
    GOAL: Verify serve method handles partial parameter overrides correctly.

    Tests that when only some parameters are overridden, the non-overridden
    parameters retain their original values.
    """
    # Create an app environment with multiple inputs
    app_env = AppEnvironment(
        name="partial-override-app",
        image=Image.from_base("python:3.11"),
            parameters=[
            Parameter(value="original-config.yaml", name="config"),
            Parameter(value="original-model.pkl", name="model"),
            Parameter(value="original-data.csv", name="data"),
        ],
    )

    # Only override the "model" input
    input_values = {
        "partial-override-app": {
            "model": "new-model.pkl",
        }
    }

    serve = _Serve(input_values=input_values)

    # Test the override extraction logic
    app_env_input_values = serve._input_values.get(app_env.name)
    assert app_env_input_values is not None

    parameter_overrides = []
    for _input in app_env.inputs:
        value = app_env_input_values.get(_input.name, _input.value)
        parameter_overrides.append(replace(_input, value=value))

    # Verify partial overrides
    assert parameter_overrides[0].value == "original-config.yaml"  # Not overridden
    assert parameter_overrides[1].value == "new-model.pkl"  # Overridden
    assert parameter_overrides[2].value == "original-data.csv"  # Not overridden


@pytest.mark.asyncio
async def test_serve_with_file_dir_parameter_overrides():
    """
    GOAL: Verify serve method handles File/Dir input overrides correctly.

    Tests that File and Dir objects can be used as override values.
    """
    import flyte.io

    original_file = flyte.io.File(path="s3://original/file.txt")
    original_dir = flyte.io.Dir(path="s3://original/dir")

    app_env = AppEnvironment(
        name="file-dir-app",
        image=Image.from_base("python:3.11"),
            parameters=[
            Parameter(value=original_file, name="myfile"),
            Parameter(value=original_dir, name="mydir"),
        ],
    )

    new_file = flyte.io.File(path="s3://new/file.txt")
    new_dir = flyte.io.Dir(path="s3://new/dir")

    input_values = {
        "file-dir-app": {
            "myfile": new_file,
            "mydir": new_dir,
        }
    }

    serve = _Serve(input_values=input_values)

    # Test the override extraction logic
    app_env_input_values = serve._input_values.get(app_env.name)
    assert app_env_input_values is not None

    parameter_overrides = []
    for _input in app_env.inputs:
        value = app_env_input_values.get(_input.name, _input.value)
        parameter_overrides.append(replace(_input, value=value))

    # Verify File/Dir overrides
    assert isinstance(parameter_overrides[0].value, flyte.io.File)
    assert parameter_overrides[0].value.path == "s3://new/file.txt"
    assert isinstance(parameter_overrides[1].value, flyte.io.Dir)
    assert parameter_overrides[1].value.path == "s3://new/dir"


# =============================================================================
# Integration tests for serve context with input overrides
# =============================================================================


def test_parameter_overrides_affect_container_cmd():
    """
    GOAL: Verify that input overrides from serve context affect the container command.

    Tests the full flow: serve context -> input overrides -> container_cmd serialization.
    """
    from dataclasses import replace

    from flyte.app._parameter import SerializableParameterCollection
    from flyte.models import CodeBundle, SerializationContext

    # Create app environment
    app_env = AppEnvironment(
        name="integration-app",
        image=Image.from_base("python:3.11"),
            parameters=[
            Parameter(value="original-config.yaml", name="config"),
            Parameter(value="s3://original/data", name="data"),
        ],
    )

    # Simulate serve context with input overrides
    input_values = {
        "integration-app": {
            "config": "overridden-config.yaml",
            "data": "s3://overridden/data",
        }
    }

    serve = _Serve(input_values=input_values)

    # Extract input overrides (replicating serve method logic)
    app_env_input_values = serve._input_values.get(app_env.name)
    parameter_overrides = []
    for _input in app_env.inputs:
        value = app_env_input_values.get(_input.name, _input.value)
        parameter_overrides.append(replace(_input, value=value))

    # Create serialization context
    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=CodeBundle(computed_version="v1.0.0", tgz="s3://bucket/code.tgz"),
    )

    # Generate container command with overrides
    cmd = app_env.container_cmd(ctx, parameter_overrides=parameter_overrides)

    # Verify the command contains overridden inputs
    assert "--inputs" in cmd
    inputs_idx = cmd.index("--inputs")
    serialized = cmd[inputs_idx + 1]

    deserialized = SerializableParameterCollection.from_transport(serialized)
    assert deserialized.inputs[0].value == "overridden-config.yaml"
    assert deserialized.inputs[1].value == "s3://overridden/data"


def test_multiple_app_environments_with_different_overrides():
    """
    GOAL: Verify different apps can have different input overrides.

    Tests that when multiple apps are defined, each can have its own
    set of input overrides in the serve context.
    """
    from dataclasses import replace

    # Create multiple app environments
    app_one = AppEnvironment(
        name="app-one",
        image=Image.from_base("python:3.11"),
            parameters=[
            Parameter(value="app-one-config.yaml", name="config"),
        ],
    )

    app_two = AppEnvironment(
        name="app-two",
        image=Image.from_base("python:3.11"),
            parameters=[
            Parameter(value="app-two-config.yaml", name="config"),
        ],
    )

    # Serve context with different overrides for each app
    input_values = {
        "app-one": {"config": "app-one-override.yaml"},
        "app-two": {"config": "app-two-override.yaml"},
    }

    serve = _Serve(input_values=input_values)

    # Extract overrides for app-one
    app_one_values = serve._input_values.get(app_one.name)
    app_one_overrides = []
    for _input in app_one.inputs:
        value = app_one_values.get(_input.name, _input.value)
        app_one_overrides.append(replace(_input, value=value))

    # Extract overrides for app-two
    app_two_values = serve._input_values.get(app_two.name)
    app_two_overrides = []
    for _input in app_two.inputs:
        value = app_two_values.get(_input.name, _input.value)
        app_two_overrides.append(replace(_input, value=value))

    # Verify different overrides for each app
    assert app_one_overrides[0].value == "app-one-override.yaml"
    assert app_two_overrides[0].value == "app-two-override.yaml"


def test_with_servecontext_dependent_apps_with_parameter_overrides():
    """
    GOAL: Verify with_servecontext correctly applies input overrides to dependent apps.

    Tests that when using with_servecontext with two apps where one depends on another,
    the input_values dict correctly updates inputs for both apps.
    """
    import flyte.io
    from flyte.app._parameter import SerializableParameterCollection
    from flyte.models import CodeBundle, SerializationContext

    # Create the backend app (dependency) with multiple inputs
    backend_app = AppEnvironment(
        name="backend-api",
        image=Image.from_base("python:3.11"),
            parameters=[
            Parameter(value="postgres://localhost:5432/db", name="database_url"),
            Parameter(value="redis://localhost:6379", name="cache_url"),
            Parameter(value=flyte.io.File(path="s3://bucket/model.pkl"), name="model_file"),
        ],
    )

    # Create the frontend app that depends on the backend
    frontend_app = AppEnvironment(
        name="frontend-app",
        image=Image.from_base("python:3.11"),
        depends_on=[backend_app],
            parameters=[
            Parameter(value="http://localhost:8000", name="api_endpoint"),
            Parameter(value="default-theme", name="theme"),
            Parameter(value=flyte.io.Dir(path="s3://bucket/assets"), name="static_assets"),
        ],
    )

    # Use with_servecontext with input overrides for BOTH apps
    new_model_file = flyte.io.File(path="s3://bucket/production-model.pkl")
    new_assets_dir = flyte.io.Dir(path="s3://bucket/production-assets")

    serve = with_servecontext(
        version="v1.0.0",
        project="production",
        domain="prod",
        input_values={
            backend_app.name: {
                "database_url": "postgres://prod-db:5432/production",
                "cache_url": "redis://prod-cache:6379",
                "model_file": new_model_file,
            },
            frontend_app.name: {
                "api_endpoint": "https://api.production.example.com",
                "theme": "production-theme",
                "static_assets": new_assets_dir,
            },
        },
    )

    # Verify serve context was created with correct parameters
    assert serve._version == "v1.0.0"
    assert serve._project == "production"
    assert serve._domain == "prod"

    # Extract and verify overrides for backend app
    backend_input_values = serve._input_values.get(backend_app.name)
    assert backend_input_values is not None

    backend_overrides = []
    for _input in backend_app.inputs:
        value = backend_input_values.get(_input.name, _input.value)
        backend_overrides.append(replace(_input, value=value))

    assert len(backend_overrides) == 3
    assert backend_overrides[0].name == "database_url"
    assert backend_overrides[0].value == "postgres://prod-db:5432/production"
    assert backend_overrides[1].name == "cache_url"
    assert backend_overrides[1].value == "redis://prod-cache:6379"
    assert backend_overrides[2].name == "model_file"
    assert isinstance(backend_overrides[2].value, flyte.io.File)
    assert backend_overrides[2].value.path == "s3://bucket/production-model.pkl"

    # Extract and verify overrides for frontend app (which depends on backend)
    frontend_input_values = serve._input_values.get(frontend_app.name)
    assert frontend_input_values is not None

    frontend_overrides = []
    for _input in frontend_app.inputs:
        value = frontend_input_values.get(_input.name, _input.value)
        frontend_overrides.append(replace(_input, value=value))

    assert len(frontend_overrides) == 3
    assert frontend_overrides[0].name == "api_endpoint"
    assert frontend_overrides[0].value == "https://api.production.example.com"
    assert frontend_overrides[1].name == "theme"
    assert frontend_overrides[1].value == "production-theme"
    assert frontend_overrides[2].name == "static_assets"
    assert isinstance(frontend_overrides[2].value, flyte.io.Dir)
    assert frontend_overrides[2].value.path == "s3://bucket/production-assets"

    # Verify that the dependency relationship is preserved
    assert backend_app in frontend_app.depends_on

    # Verify container_cmd serialization works correctly for both apps
    ctx = SerializationContext(
        org="test-org",
        project="production",
        domain="prod",
        version="v1.0.0",
        code_bundle=CodeBundle(computed_version="v1.0.0", tgz="s3://bucket/code.tgz"),
    )

    # Check backend app command serialization
    backend_cmd = backend_app.container_cmd(ctx, parameter_overrides=backend_overrides)
    assert "--inputs" in backend_cmd
    backend_inputs_idx = backend_cmd.index("--inputs")
    backend_serialized = backend_cmd[backend_inputs_idx + 1]
    backend_deserialized = SerializableParameterCollection.from_transport(backend_serialized)
    assert backend_deserialized.inputs[0].value == "postgres://prod-db:5432/production"
    assert backend_deserialized.inputs[1].value == "redis://prod-cache:6379"

    # Check frontend app command serialization
    frontend_cmd = frontend_app.container_cmd(ctx, parameter_overrides=frontend_overrides)
    assert "--inputs" in frontend_cmd
    frontend_inputs_idx = frontend_cmd.index("--inputs")
    frontend_serialized = frontend_cmd[frontend_inputs_idx + 1]
    frontend_deserialized = SerializableParameterCollection.from_transport(frontend_serialized)
    assert frontend_deserialized.inputs[0].value == "https://api.production.example.com"
    assert frontend_deserialized.inputs[1].value == "production-theme"
