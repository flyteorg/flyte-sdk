"""
Unit tests for flyte._serve module.

These tests verify the serve context functionality including:
- _Serve class initialization and configuration
- with_servecontext() function
- Parameter value override handling
- Local serving mode
"""

import json
import pathlib
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from flyte._image import Image
from flyte._serve import _LOCAL_APP_ENDPOINTS, _LocalApp, _Serve, with_servecontext
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
    assert serve._parameter_values == {}
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
        parameter_values={"my-app": {"config": "new-config.yaml"}},
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
    assert serve._parameter_values == {"my-app": {"config": "new-config.yaml"}}
    assert serve._cluster_pool == "gpu-pool"
    assert serve._log_level == 10
    assert serve._log_format == "json"


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
        parameter_values={"app": {"key": "value"}},
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
    assert serve._parameter_values == {"app": {"key": "value"}}
    assert serve._cluster_pool == "cpu-pool"
    assert serve._log_level == 20
    assert serve._log_format == "json"


# =============================================================================
# Tests for parameter value override functionality in _Serve
# =============================================================================


def test_parameter_values_dict_structure():
    """
    GOAL: Verify parameter_values dict has correct structure.

    Tests that parameter_values maps app environment names to dicts of parameter names to values.
    """
    import flyte.io

    parameter_values = {
        "app-one": {
            "config": "config-one.yaml",
            "data": flyte.io.Dir(path="s3://bucket/data-one"),
        },
        "app-two": {
            "model": flyte.io.File(path="s3://bucket/model.pkl"),
        },
    }

    serve = _Serve(parameter_values=parameter_values)

    assert "app-one" in serve._parameter_values
    assert "app-two" in serve._parameter_values
    assert serve._parameter_values["app-one"]["config"] == "config-one.yaml"
    assert isinstance(serve._parameter_values["app-one"]["data"], flyte.io.Dir)
    assert isinstance(serve._parameter_values["app-two"]["model"], flyte.io.File)


@pytest.mark.asyncio
async def test_serve_extracts_parameter_overrides_for_matching_app():
    """
    GOAL: Verify serve method correctly extracts parameter overrides for matching app.

    Tests that when parameter_values contains entries for the app being served,
    the overrides are correctly extracted and applied.
    """

    # Create an app environment with parameters
    app_env = AppEnvironment(
        name="my-test-app",
        image=Image.from_base("python:3.11"),
        parameters=[
            Parameter(value="original-config.yaml", name="config"),
            Parameter(value="s3://original/data", name="data"),
        ],
    )

    # Create serve context with parameter overrides for this app
    parameter_values = {
        "my-test-app": {
            "config": "overridden-config.yaml",
            "data": "s3://new/data",
        }
    }

    serve = _Serve(parameter_values=parameter_values)

    # Manually test the override extraction logic (from _Serve.serve method)
    app_env_parameter_values = serve._parameter_values.get(app_env.name)
    assert app_env_parameter_values is not None

    parameter_overrides = []
    for parameter in app_env.parameters:
        value = app_env_parameter_values.get(parameter.name, parameter.value)
        parameter_overrides.append(replace(parameter, value=value))

    # Verify overrides were created correctly
    assert len(parameter_overrides) == 2
    assert parameter_overrides[0].name == "config"
    assert parameter_overrides[0].value == "overridden-config.yaml"
    assert parameter_overrides[1].name == "data"
    assert parameter_overrides[1].value == "s3://new/data"


@pytest.mark.asyncio
async def test_serve_no_overrides_for_non_matching_app():
    """
    GOAL: Verify serve method returns None for apps not in parameter_values.

    Tests that when parameter_values doesn't contain entries for the app being served,
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

    # Create serve context with parameter overrides for a DIFFERENT app
    parameter_values = {
        "other-app": {
            "config": "other-config.yaml",
        }
    }

    serve = _Serve(parameter_values=parameter_values)

    # Test the override extraction logic
    app_env_parameter_values = serve._parameter_values.get(app_env.name)
    assert app_env_parameter_values is None  # No overrides for this app


@pytest.mark.asyncio
async def test_serve_partial_parameter_overrides():
    """
    GOAL: Verify serve method handles partial parameter overrides correctly.

    Tests that when only some parameters are overridden, the non-overridden
    parameters retain their original values.
    """
    # Create an app environment with multiple parameters
    app_env = AppEnvironment(
        name="partial-override-app",
        image=Image.from_base("python:3.11"),
        parameters=[
            Parameter(value="original-config.yaml", name="config"),
            Parameter(value="original-model.pkl", name="model"),
            Parameter(value="original-data.csv", name="data"),
        ],
    )

    # Only override the "model" parameter
    parameter_values = {
        "partial-override-app": {
            "model": "new-model.pkl",
        }
    }

    serve = _Serve(parameter_values=parameter_values)

    # Test the override extraction logic
    app_env_parameter_values = serve._parameter_values.get(app_env.name)
    assert app_env_parameter_values is not None

    parameter_overrides = []
    for _parameter in app_env.parameters:
        value = app_env_parameter_values.get(_parameter.name, _parameter.value)
        parameter_overrides.append(replace(_parameter, value=value))

    # Verify partial overrides
    assert parameter_overrides[0].value == "original-config.yaml"  # Not overridden
    assert parameter_overrides[1].value == "new-model.pkl"  # Overridden
    assert parameter_overrides[2].value == "original-data.csv"  # Not overridden


@pytest.mark.asyncio
async def test_serve_with_file_dir_parameter_overrides():
    """
    GOAL: Verify serve method handles File/Dir parameter overrides correctly.

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

    parameter_values = {
        "file-dir-app": {
            "myfile": new_file,
            "mydir": new_dir,
        }
    }

    serve = _Serve(parameter_values=parameter_values)

    # Test the override extraction logic
    app_env_parameter_values = serve._parameter_values.get(app_env.name)
    assert app_env_parameter_values is not None

    parameter_overrides = []
    for _parameter in app_env.parameters:
        value = app_env_parameter_values.get(_parameter.name, _parameter.value)
        parameter_overrides.append(replace(_parameter, value=value))

    # Verify File/Dir overrides
    assert isinstance(parameter_overrides[0].value, flyte.io.File)
    assert parameter_overrides[0].value.path == "s3://new/file.txt"
    assert isinstance(parameter_overrides[1].value, flyte.io.Dir)
    assert parameter_overrides[1].value.path == "s3://new/dir"


# =============================================================================
# Integration tests for serve context with parameter overrides
# =============================================================================


def test_parameter_overrides_affect_container_cmd():
    """
    GOAL: Verify that parameter overrides from serve context affect the container command.

    Tests the full flow: serve context -> parameter overrides -> container_cmd serialization.
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

    # Simulate serve context with parameter overrides
    parameter_values = {
        "integration-app": {
            "config": "overridden-config.yaml",
            "data": "s3://overridden/data",
        }
    }

    serve = _Serve(parameter_values=parameter_values)

    # Extract parameter overrides (replicating serve method logic)
    app_env_parameter_values = serve._parameter_values.get(app_env.name)
    parameter_overrides = []
    for _parameter in app_env.parameters:
        value = app_env_parameter_values.get(_parameter.name, _parameter.value)
        parameter_overrides.append(replace(_parameter, value=value))

    # Create serialization context
    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=CodeBundle(computed_version="v1.0.0", tgz="s3://bucket/code.tgz"),
        root_dir=pathlib.Path.cwd(),
    )

    # Generate container command with overrides
    cmd = app_env.container_cmd(ctx, parameter_overrides=parameter_overrides)

    # Verify the command contains overridden parameters
    assert "--parameters" in cmd
    parameters_idx = cmd.index("--parameters")
    serialized = cmd[parameters_idx + 1]

    deserialized = SerializableParameterCollection.from_transport(serialized)
    assert deserialized.parameters[0].value == "overridden-config.yaml"
    assert deserialized.parameters[1].value == "s3://overridden/data"


def test_multiple_app_environments_with_different_overrides():
    """
    GOAL: Verify different apps can have different parameter overrides.

    Tests that when multiple apps are defined, each can have its own
    set of parameter overrides in the serve context.
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

    # Serve context with different parameter overrides for each app
    parameter_values = {
        "app-one": {"config": "app-one-override.yaml"},
        "app-two": {"config": "app-two-override.yaml"},
    }

    serve = _Serve(parameter_values=parameter_values)

    # Extract overrides for app-one
    app_one_values = serve._parameter_values.get(app_one.name)
    app_one_overrides = []
    for _parameter in app_one.parameters:
        value = app_one_values.get(_parameter.name, _parameter.value)
        app_one_overrides.append(replace(_parameter, value=value))

    # Extract overrides for app-two
    app_two_values = serve._parameter_values.get(app_two.name)
    app_two_overrides = []
    for _parameter in app_two.parameters:
        value = app_two_values.get(_parameter.name, _parameter.value)
        app_two_overrides.append(replace(_parameter, value=value))

    # Verify different overrides for each app
    assert app_one_overrides[0].value == "app-one-override.yaml"
    assert app_two_overrides[0].value == "app-two-override.yaml"


def test_with_servecontext_dependent_apps_with_parameter_overrides():
    """
    GOAL: Verify with_servecontext correctly applies parameter overrides to dependent apps.

    Tests that when using with_servecontext with two apps where one depends on another,
    the parameter_values dict correctly updates parameters for both apps.
    """
    import flyte.io
    from flyte.app._parameter import SerializableParameterCollection
    from flyte.models import CodeBundle, SerializationContext

    # Create the backend app (dependency) with multiple parameters
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

    # Use with_servecontext with parameter overrides for BOTH apps
    new_model_file = flyte.io.File(path="s3://bucket/production-model.pkl")
    new_assets_dir = flyte.io.Dir(path="s3://bucket/production-assets")

    serve = with_servecontext(
        version="v1.0.0",
        project="production",
        domain="prod",
        parameter_values={
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
    backend_parameter_values = serve._parameter_values.get(backend_app.name)
    assert backend_parameter_values is not None

    backend_overrides = []
    for _parameter in backend_app.parameters:
        value = backend_parameter_values.get(_parameter.name, _parameter.value)
        backend_overrides.append(replace(_parameter, value=value))

    assert len(backend_overrides) == 3
    assert backend_overrides[0].name == "database_url"
    assert backend_overrides[0].value == "postgres://prod-db:5432/production"
    assert backend_overrides[1].name == "cache_url"
    assert backend_overrides[1].value == "redis://prod-cache:6379"
    assert backend_overrides[2].name == "model_file"
    assert isinstance(backend_overrides[2].value, flyte.io.File)
    assert backend_overrides[2].value.path == "s3://bucket/production-model.pkl"

    # Extract and verify overrides for frontend app (which depends on backend)
    frontend_parameter_values = serve._parameter_values.get(frontend_app.name)
    assert frontend_parameter_values is not None

    frontend_overrides = []
    for _parameter in frontend_app.parameters:
        value = frontend_parameter_values.get(_parameter.name, _parameter.value)
        frontend_overrides.append(replace(_parameter, value=value))

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
        root_dir=pathlib.Path.cwd(),
    )

    # Check backend app command serialization
    backend_cmd = backend_app.container_cmd(ctx, parameter_overrides=backend_overrides)
    assert "--parameters" in backend_cmd
    backend_parameters_idx = backend_cmd.index("--parameters")
    backend_serialized = backend_cmd[backend_parameters_idx + 1]
    backend_deserialized = SerializableParameterCollection.from_transport(backend_serialized)
    assert backend_deserialized.parameters[0].value == "postgres://prod-db:5432/production"
    assert backend_deserialized.parameters[1].value == "redis://prod-cache:6379"

    # Check frontend app command serialization
    frontend_cmd = frontend_app.container_cmd(ctx, parameter_overrides=frontend_overrides)
    assert "--parameters" in frontend_cmd
    frontend_parameters_idx = frontend_cmd.index("--parameters")
    frontend_serialized = frontend_cmd[frontend_parameters_idx + 1]
    frontend_deserialized = SerializableParameterCollection.from_transport(frontend_serialized)
    assert frontend_deserialized.parameters[0].value == "https://api.production.example.com"
    assert frontend_deserialized.parameters[1].value == "production-theme"


# =============================================================================
# Tests for serve method deploying dependent apps
# =============================================================================


def test_serve_discovers_dependent_app_environments():
    """
    GOAL: Verify plan_deploy discovers dependent AppEnvironments.

    Tests that when an app depends on another app, plan_deploy correctly
    discovers both apps in the deployment plan.
    """
    from flyte._deploy import plan_deploy

    # Create the backend app (dependency)
    backend_app = AppEnvironment(
        name="backend-api",
        image=Image.from_base("python:3.11"),
    )

    # Create the frontend app that depends on the backend
    frontend_app = AppEnvironment(
        name="frontend-app",
        image=Image.from_base("python:3.11"),
        depends_on=[backend_app],
    )

    # Plan deployment for frontend app
    deployments = plan_deploy(frontend_app)

    # Verify both apps are in the deployment plan
    assert len(deployments) == 1
    deployment_plan = deployments[0]
    assert "backend-api" in deployment_plan.envs
    assert "frontend-app" in deployment_plan.envs
    assert deployment_plan.envs["backend-api"] is backend_app
    assert deployment_plan.envs["frontend-app"] is frontend_app


def test_serve_discovers_transitive_dependencies():
    """
    GOAL: Verify plan_deploy discovers transitive dependencies.

    Tests that when app C depends on app B, which depends on app A,
    all three apps are discovered in the deployment plan.
    """
    from flyte._deploy import plan_deploy

    # Create chain of dependencies: app_c -> app_b -> app_a
    app_a = AppEnvironment(
        name="app-a",
        image=Image.from_base("python:3.11"),
    )

    app_b = AppEnvironment(
        name="app-b",
        image=Image.from_base("python:3.11"),
        depends_on=[app_a],
    )

    app_c = AppEnvironment(
        name="app-c",
        image=Image.from_base("python:3.11"),
        depends_on=[app_b],
    )

    # Plan deployment for app_c (which has transitive dependencies)
    deployments = plan_deploy(app_c)

    # Verify all three apps are in the deployment plan
    assert len(deployments) == 1
    deployment_plan = deployments[0]
    assert "app-a" in deployment_plan.envs
    assert "app-b" in deployment_plan.envs
    assert "app-c" in deployment_plan.envs


def test_serve_discovers_multiple_direct_dependencies():
    """
    GOAL: Verify plan_deploy discovers multiple direct dependencies.

    Tests that when an app depends on multiple other apps,
    all dependencies are discovered in the deployment plan.
    """
    from flyte._deploy import plan_deploy

    # Create multiple dependencies
    db_app = AppEnvironment(
        name="db-service",
        image=Image.from_base("python:3.11"),
    )

    cache_app = AppEnvironment(
        name="cache-service",
        image=Image.from_base("python:3.11"),
    )

    auth_app = AppEnvironment(
        name="auth-service",
        image=Image.from_base("python:3.11"),
    )

    # Main app depends on all three services
    main_app = AppEnvironment(
        name="main-app",
        image=Image.from_base("python:3.11"),
        depends_on=[db_app, cache_app, auth_app],
    )

    # Plan deployment for main_app
    deployments = plan_deploy(main_app)

    # Verify all four apps are in the deployment plan
    assert len(deployments) == 1
    deployment_plan = deployments[0]
    assert len(deployment_plan.envs) == 4
    assert "db-service" in deployment_plan.envs
    assert "cache-service" in deployment_plan.envs
    assert "auth-service" in deployment_plan.envs
    assert "main-app" in deployment_plan.envs


# =============================================================================
# Tests for local serving mode
# =============================================================================


def test_serve_local_mode_initialization():
    """
    GOAL: Verify _Serve initializes with mode='local' correctly.
    """
    serve = _Serve(mode="local")
    assert serve._mode == "local"


def test_serve_remote_mode_initialization():
    """
    GOAL: Verify _Serve initializes with mode='remote' when explicitly set.
    """
    serve = _Serve(mode="remote")
    assert serve._mode == "remote"


def test_with_servecontext_local_mode():
    """
    GOAL: Verify with_servecontext passes mode parameter.
    """
    serve = with_servecontext(mode="local")
    assert serve._mode == "local"


def test_with_servecontext_remote_mode():
    """
    GOAL: Verify with_servecontext passes mode='remote'.
    """
    serve = with_servecontext(mode="remote")
    assert serve._mode == "remote"


def test_local_app_properties():
    """
    GOAL: Verify _LocalApp has correct properties.
    """
    app_env = AppEnvironment(
        name="test-local-props",
        image=Image.from_base("python:3.11"),
        port=9999,
    )
    serve_obj = _Serve(mode="local")
    local_app = _LocalApp(app_env=app_env, host="127.0.0.1", port=9999, _serve_obj=serve_obj)
    assert local_app.name == "test-local-props"
    assert local_app.endpoint == "http://127.0.0.1:9999"
    assert local_app.url == "http://127.0.0.1:9999"


def test_local_serve_with_server_decorator():
    """
    GOAL: Verify local serving works with the @app_env.server decorator pattern.

    Tests that:
    - The app starts in a background thread
    - The endpoint is registered in _LOCAL_APP_ENDPOINTS
    - The endpoint is reachable
    - local_app.endpoint returns the local endpoint
    - The app can be shut down cleanly
    """

    class TestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok"}).encode())
            elif parsed.path == "/":
                params = parse_qs(parsed.query)
                x = int(params.get("x", [0])[0])
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"result": x + 1}).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass

    app_env = AppEnvironment(
        name="test-local-server-decorator",
        image=Image.from_base("python:3.11"),
        port=18091,
    )

    @app_env.server
    def serve():
        server = HTTPServer(("127.0.0.1", 18091), TestHandler)
        server.serve_forever()

    try:
        local_app = with_servecontext(mode="local").serve(app_env)

        # Verify endpoint is registered
        assert "test-local-server-decorator" in _LOCAL_APP_ENDPOINTS
        assert _LOCAL_APP_ENDPOINTS["test-local-server-decorator"] == "http://localhost:18091"

        # Verify local_app.endpoint returns the local endpoint
        # Note: Use local_app.endpoint (not app_env.endpoint) from the main thread
        assert local_app.endpoint == "http://localhost:18091"

        # Wait for readiness
        local_app.activate(wait=True)

        # Verify the endpoint is actually working
        resp = httpx.get("http://127.0.0.1:18091/", params={"x": 5})
        assert resp.status_code == 200
        assert resp.json()["result"] == 6
    finally:
        local_app.deactivate()
        # Verify endpoint is removed after shutdown
        assert "test-local-server-decorator" not in _LOCAL_APP_ENDPOINTS


def test_local_serve_with_command():
    """
    GOAL: Verify local serving works with the command specification pattern.

    Tests that:
    - The app starts as a subprocess
    - The endpoint is registered in _LOCAL_APP_ENDPOINTS
    - The endpoint is reachable
    - The app can be shut down cleanly
    """
    app_env = AppEnvironment(
        name="test-local-cmd",
        image=Image.from_base("python:3.11"),
        command="python -m http.server 18192",
        port=18192,
    )

    try:
        local_app = with_servecontext(mode="local", activate_timeout=10.0, health_check_path="/").serve(app_env)

        # Verify endpoint is registered
        assert "test-local-cmd" in _LOCAL_APP_ENDPOINTS

        # Wait for readiness
        local_app.activate(wait=True)

        # Verify the endpoint is actually working
        resp = httpx.get("http://localhost:18192/")
        assert resp.status_code == 200
    finally:
        local_app.deactivate()
        assert "test-local-cmd" not in _LOCAL_APP_ENDPOINTS


def test_local_serve_no_server_or_command_raises():
    """
    GOAL: Verify that serving an app without server function or command raises ValueError.
    """
    app_env = AppEnvironment(
        name="test-local-no-server",
        image=Image.from_base("python:3.11"),
    )

    with pytest.raises(ValueError, match="has no server function, command, or args defined"):
        with_servecontext(mode="local").serve(app_env)


def test_local_app_is_ready_timeout():
    """
    GOAL: Verify is_active returns False when app is not reachable.
    """
    app_env = AppEnvironment(
        name="test-timeout",
        image=Image.from_base("python:3.11"),
        port=18099,
    )
    serve_obj = _Serve(mode="local")
    local_app = _LocalApp(app_env=app_env, host="127.0.0.1", port=18099, _serve_obj=serve_obj)

    # Nothing listening on port 18099, so is_active should return False
    assert not local_app.is_active()


def test_local_serve_env_vars():
    """
    GOAL: Verify that env_vars from servecontext are set in the environment.
    """
    import os

    class TestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            env_val = os.environ.get("TEST_LOCAL_SERVE_VAR", "")
            self.wfile.write(json.dumps({"env_val": env_val}).encode())

        def log_message(self, format, *args):
            pass

    app_env = AppEnvironment(
        name="test-local-envvars",
        image=Image.from_base("python:3.11"),
        port=18093,
    )

    @app_env.server
    def serve():
        server = HTTPServer(("127.0.0.1", 18093), TestHandler)
        server.serve_forever()

    try:
        local_app = with_servecontext(
            mode="local",
            env_vars={"TEST_LOCAL_SERVE_VAR": "hello-world"},
        ).serve(app_env)

        local_app.activate(wait=True)

        # Verify the environment variable was set
        assert os.environ.get("TEST_LOCAL_SERVE_VAR") == "hello-world"
    finally:
        local_app.deactivate()
        os.environ.pop("TEST_LOCAL_SERVE_VAR", None)


def test_local_app_registered_in_active_set():
    """
    GOAL: Verify that creating a _LocalApp registers it in _ACTIVE_LOCAL_APPS
    and deactivating it removes it.
    """
    from flyte._serve import _ACTIVE_LOCAL_APPS

    app_env = AppEnvironment(
        name="test-active-set",
        image=Image.from_base("python:3.11"),
        port=18096,
    )
    serve_obj = _Serve(mode="local")
    local_app = _LocalApp(app_env=app_env, host="127.0.0.1", port=18096, _serve_obj=serve_obj)

    # Should be registered on creation
    assert local_app in _ACTIVE_LOCAL_APPS

    # Deactivate should remove it
    local_app.deactivate()
    assert local_app not in _ACTIVE_LOCAL_APPS


def test_cleanup_local_apps_deactivates_all():
    """
    GOAL: Verify _cleanup_local_apps() deactivates all registered _LocalApp instances.
    """
    from flyte._serve import _ACTIVE_LOCAL_APPS, _cleanup_local_apps

    app_env_a = AppEnvironment(
        name="test-cleanup-a",
        image=Image.from_base("python:3.11"),
        port=18096,
    )
    app_env_b = AppEnvironment(
        name="test-cleanup-b",
        image=Image.from_base("python:3.11"),
        port=18097,
    )
    serve_obj = _Serve(mode="local")
    local_app_a = _LocalApp(app_env=app_env_a, host="127.0.0.1", port=18096, _serve_obj=serve_obj)
    local_app_b = _LocalApp(app_env=app_env_b, host="127.0.0.1", port=18097, _serve_obj=serve_obj)

    assert local_app_a in _ACTIVE_LOCAL_APPS
    assert local_app_b in _ACTIVE_LOCAL_APPS

    _cleanup_local_apps()

    assert local_app_a not in _ACTIVE_LOCAL_APPS
    assert local_app_b not in _ACTIVE_LOCAL_APPS


def test_cleanup_terminates_subprocess():
    """
    GOAL: Verify _cleanup_local_apps() calls terminate() on child processes of registered apps.
    """
    from unittest.mock import MagicMock

    from flyte._serve import _ACTIVE_LOCAL_APPS, _cleanup_local_apps

    app_env = AppEnvironment(
        name="test-cleanup-proc",
        image=Image.from_base("python:3.11"),
        port=18098,
    )

    serve_obj = _Serve(mode="local")
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # simulate a running process
    local_app = _LocalApp(app_env=app_env, host="127.0.0.1", port=18098, _serve_obj=serve_obj, process=mock_proc)

    assert local_app in _ACTIVE_LOCAL_APPS

    _cleanup_local_apps()

    # The process should have been asked to terminate
    mock_proc.terminate.assert_called_once()
    assert local_app not in _ACTIVE_LOCAL_APPS


def test_signal_handler_cleans_up_apps():
    """
    GOAL: Verify that simulating SIGINT triggers cleanup of registered local apps.
    """
    import signal as _signal

    from flyte._serve import (
        _ACTIVE_LOCAL_APPS,
        _install_signal_handlers,
        _signal_handler,
    )

    # Ensure signal handlers are installed
    _install_signal_handlers()

    app_env = AppEnvironment(
        name="test-signal-cleanup",
        image=Image.from_base("python:3.11"),
        port=18096,
    )
    serve_obj = _Serve(mode="local")
    local_app = _LocalApp(app_env=app_env, host="127.0.0.1", port=18096, _serve_obj=serve_obj)
    assert local_app in _ACTIVE_LOCAL_APPS

    # Directly invoke the signal handler (don't actually send a signal
    # because that would interrupt the test runner).  We pass frame=None.
    # Patch the original handler to avoid side-effects.
    with patch("flyte._serve._ORIGINAL_SIGINT_HANDLER", new=_signal.SIG_IGN):
        _signal_handler(_signal.SIGINT, None)

    assert local_app not in _ACTIVE_LOCAL_APPS


def test_deactivate_idempotent():
    """
    GOAL: Verify that calling deactivate() multiple times is safe.
    """
    from flyte._serve import _ACTIVE_LOCAL_APPS

    app_env = AppEnvironment(
        name="test-deactivate-idempotent",
        image=Image.from_base("python:3.11"),
        port=18096,
    )
    serve_obj = _Serve(mode="local")
    local_app = _LocalApp(app_env=app_env, host="127.0.0.1", port=18096, _serve_obj=serve_obj)
    assert local_app in _ACTIVE_LOCAL_APPS

    local_app.deactivate()
    assert local_app not in _ACTIVE_LOCAL_APPS

    # Second call should not raise
    local_app.deactivate()
    assert local_app not in _ACTIVE_LOCAL_APPS


def test_local_serve_endpoint_resolves_correctly():
    """
    GOAL: Verify that app_env.endpoint resolves to the local endpoint when the app
    is registered in _LOCAL_APP_ENDPOINTS (i.e., when served locally in-process).

    This is the key integration point: when an AppEnvironment is served locally,
    its .endpoint property should return the local address so other code in the
    same process can call it.
    """
    from flyte._serve import _LOCAL_APP_ENDPOINTS

    app_env = AppEnvironment(
        name="test-endpoint-resolve",
        image=Image.from_base("python:3.11"),
        port=18094,
    )

    # Simulate what _serve_local_with_server_func does: register the endpoint
    _LOCAL_APP_ENDPOINTS[app_env.name] = f"http://localhost:{app_env.port.port}"
    try:
        # The key assertion: app_env.endpoint should resolve to the registered local endpoint
        endpoint = app_env.endpoint
        assert endpoint == "http://localhost:18094"
    finally:
        # Clean up
        del _LOCAL_APP_ENDPOINTS[app_env.name]


def test_local_serve_endpoint_resolves_via_env_var():
    """
    GOAL: Verify that app_env.endpoint resolves to the local endpoint when
    _FSERVE_MODE env var is set to "local" (for subprocess-based apps).
    """
    import os

    from flyte._serve import _FSERVE_MODE_ENV_VAR

    app_env = AppEnvironment(
        name="test-endpoint-resolve-env",
        image=Image.from_base("python:3.11"),
        port=18095,
    )

    # Simulate subprocess environment
    os.environ[_FSERVE_MODE_ENV_VAR] = "local"
    try:
        endpoint = app_env.endpoint
        assert endpoint == "http://localhost:18095"
    finally:
        del os.environ[_FSERVE_MODE_ENV_VAR]


def test_local_serve_endpoint_resolves_via_context_var():
    """
    GOAL: Verify that app_env.endpoint resolves to the local endpoint when
    serve_mode_var context variable is set to "local".

    This tests the direct context variable access without going through
    the full serve machinery.
    """
    from flyte._serve import serve_mode_var

    app_env = AppEnvironment(
        name="test-endpoint-resolve-ctx-var",
        image=Image.from_base("python:3.11"),
        port=18096,
    )

    # Set the context variable directly
    token = serve_mode_var.set("local")
    try:
        endpoint = app_env.endpoint
        assert endpoint == "http://localhost:18096"
    finally:
        serve_mode_var.reset(token)


def test_serve_mode_var_propagates_to_async_tasks():
    """
    GOAL: Verify that serve_mode_var propagates to async tasks created via
    the custom task factory installed by _serve_local_with_server_func.

    This tests the core mechanism: when we set serve_mode_var and install
    a custom task factory, all tasks created in that event loop should
    inherit the context with serve_mode_var set to "local".
    """
    import asyncio
    import contextvars

    from flyte._serve import serve_mode_var

    results = []

    async def check_serve_mode():
        """Coroutine that checks the serve_mode_var value."""
        results.append(serve_mode_var.get())

    async def spawn_task_and_check():
        """Spawn a new task (simulating what uvicorn does) and check serve_mode."""
        # This simulates what an ASGI server does: create a new task for each request
        task = asyncio.create_task(check_serve_mode())
        await task

    def run_with_context_task_factory():
        """Run the test with the same pattern used in _serve_local_with_server_func."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Set the context variable
        serve_mode_var.set("local")

        # Copy the context after setting serve_mode_var
        ctx = contextvars.copy_context()

        # Install the same custom task factory used in _serve_local_with_server_func
        def _context_task_factory(loop, coro, context=None):
            return asyncio.Task(coro, loop=loop, context=context or ctx)

        loop.set_task_factory(_context_task_factory)

        try:
            # Run a coroutine that spawns a task (like uvicorn handling a request)
            loop.run_until_complete(spawn_task_and_check())
        finally:
            loop.close()

    # Run in a separate thread to avoid interfering with the test's event loop
    import threading

    thread = threading.Thread(target=run_with_context_task_factory)
    thread.start()
    thread.join(timeout=5)

    # The task should have seen serve_mode_var as "local"
    assert len(results) == 1
    assert results[0] == "local"


def test_serve_mode_var_default_is_remote():
    """
    GOAL: Verify that serve_mode_var defaults to "remote".
    """
    # In a fresh context, the default should be "remote"
    import contextvars

    from flyte._serve import serve_mode_var

    ctx = contextvars.copy_context()
    value = ctx.run(serve_mode_var.get)
    assert value == "remote"


# =============================================================================
# Tests for is_deactivated() thread-aware fix
# =============================================================================


def test_is_deactivated_false_while_thread_alive():
    """
    GOAL: Verify is_deactivated() returns False while the server thread is alive.

    Previously, is_deactivated() only checked self._process and would always
    return True for thread-based apps (since _process is None).
    """
    import threading

    app_env = AppEnvironment(
        name="test-deactivated-thread-alive",
        image=Image.from_base("python:3.11"),
        port=18200,
    )

    stop = threading.Event()

    def _blocker():
        stop.wait()

    serve_obj = _Serve(mode="local")
    thread = threading.Thread(target=_blocker, daemon=True)
    thread.start()

    local_app = _LocalApp(app_env=app_env, host="127.0.0.1", port=18200, _serve_obj=serve_obj, thread=thread)

    try:
        # Thread is alive, so is_deactivated should return False
        assert not local_app.is_deactivated()
    finally:
        stop.set()
        thread.join(timeout=2)
        local_app.deactivate()


def test_is_deactivated_true_after_thread_exits():
    """
    GOAL: Verify is_deactivated() returns True after the server thread has exited.
    """
    import threading

    app_env = AppEnvironment(
        name="test-deactivated-thread-exited",
        image=Image.from_base("python:3.11"),
        port=18201,
    )

    def _noop():
        pass

    serve_obj = _Serve(mode="local")
    thread = threading.Thread(target=_noop, daemon=True)
    thread.start()
    thread.join(timeout=2)  # Wait for it to finish

    local_app = _LocalApp(app_env=app_env, host="127.0.0.1", port=18201, _serve_obj=serve_obj, thread=thread)

    try:
        # Thread has exited, so is_deactivated should return True
        assert local_app.is_deactivated()
    finally:
        local_app.deactivate()


def test_is_deactivated_true_no_thread_no_process():
    """
    GOAL: Verify is_deactivated() returns True when neither thread nor process is set.
    """
    app_env = AppEnvironment(
        name="test-deactivated-no-thread-no-proc",
        image=Image.from_base("python:3.11"),
        port=18202,
    )
    serve_obj = _Serve(mode="local")
    local_app = _LocalApp(app_env=app_env, host="127.0.0.1", port=18202, _serve_obj=serve_obj)

    try:
        assert local_app.is_deactivated()
    finally:
        local_app.deactivate()


# =============================================================================
# Tests for thread-based deactivate() with stop_event and loop shutdown
# =============================================================================


def test_deactivate_stops_thread_based_app():
    """
    GOAL: Verify deactivate(wait=True) stops a thread-based app by signalling
    the stop event and stopping the event loop.
    """
    import asyncio
    import threading

    app_env = AppEnvironment(
        name="test-deactivate-thread-app",
        image=Image.from_base("python:3.11"),
        port=18203,
    )

    serve_obj = _Serve(mode="local")
    local_app = _LocalApp(app_env=app_env, host="127.0.0.1", port=18203, _serve_obj=serve_obj)

    # Simulate the pattern used in _serve_local_with_server_func:
    # create an event loop in a thread, store it on local_app, and run forever.
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        local_app._thread_loop = loop
        try:
            loop.run_forever()
        finally:
            local_app._thread_loop = None
            loop.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    local_app._thread = thread

    # Give the thread a moment to start the loop
    import time

    time.sleep(0.1)

    assert not local_app.is_deactivated()
    assert local_app._stop_event is not None
    assert not local_app._stop_event.is_set()

    # Deactivate should set stop_event, stop the loop, and join the thread
    local_app.deactivate(wait=True)

    assert local_app._stop_event.is_set()
    assert not thread.is_alive()
    assert local_app.is_deactivated()


def test_deactivate_sets_stop_event_for_thread():
    """
    GOAL: Verify deactivate() sets the _stop_event even without wait=True.
    """
    import threading

    app_env = AppEnvironment(
        name="test-deactivate-stop-event",
        image=Image.from_base("python:3.11"),
        port=18204,
    )

    serve_obj = _Serve(mode="local")
    stop = threading.Event()

    def _blocker():
        stop.wait()

    thread = threading.Thread(target=_blocker, daemon=True)
    thread.start()

    local_app = _LocalApp(app_env=app_env, host="127.0.0.1", port=18204, _serve_obj=serve_obj, thread=thread)

    local_app.deactivate()
    assert local_app._stop_event.is_set()

    # Clean up
    stop.set()
    thread.join(timeout=2)


# =============================================================================
# Tests for AppHandle Protocol
# =============================================================================


def test_local_app_satisfies_app_handle_protocol():
    """
    GOAL: Verify that _LocalApp is recognized as an AppHandle instance.
    """
    from flyte._serve import AppHandle

    app_env = AppEnvironment(
        name="test-protocol-local",
        image=Image.from_base("python:3.11"),
        port=18205,
    )
    serve_obj = _Serve(mode="local")
    local_app = _LocalApp(app_env=app_env, host="127.0.0.1", port=18205, _serve_obj=serve_obj)

    try:
        assert isinstance(local_app, AppHandle)
    finally:
        local_app.deactivate()


def test_app_handle_protocol_has_required_attributes():
    """
    GOAL: Verify the AppHandle protocol defines the expected interface.
    """
    from flyte._serve import AppHandle

    # Check that the Protocol has the expected abstract members
    assert hasattr(AppHandle, "name")
    assert hasattr(AppHandle, "endpoint")
    assert hasattr(AppHandle, "is_active")
    assert hasattr(AppHandle, "is_deactivated")
    assert hasattr(AppHandle, "activate")
    assert hasattr(AppHandle, "deactivate")
