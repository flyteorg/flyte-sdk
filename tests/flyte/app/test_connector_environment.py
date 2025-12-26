"""
Comprehensive unit tests for ConnectorEnvironment.

These tests verify ConnectorEnvironment functionality without using mocks,
focusing on container_cmd, container_args, and connector-specific behavior.
"""

import pathlib

import pytest

from flyte._image import Image
from flyte.app import ConnectorEnvironment
from flyte.app._parameter import Parameter
from flyte.app._types import Port
from flyte.models import CodeBundle, SerializationContext


def test_connector_environment_default_values():
    """
    GOAL: Verify that ConnectorEnvironment has correct default values.

    Tests that:
    - Default type is "Connector"
    - Default port is Port(port=8080, name="h2c")
    - Other defaults are inherited from AppEnvironment
    """
    conn_env = ConnectorEnvironment(
        name="my-connector",
        image=Image.from_base("python:3.11"),
    )

    # Verify connector-specific defaults
    assert conn_env.type == "Connector"
    assert isinstance(conn_env.port, Port)
    assert conn_env.port.port == 8080
    assert conn_env.port.name == "h2c"

    # Verify inherited defaults
    assert conn_env.requires_auth is True
    assert conn_env.cluster_pool == "default"
    assert conn_env.args is None
    assert conn_env.command is None


def test_connector_environment_custom_port():
    """
    GOAL: Verify that custom ports can be set and override the default.

    Tests that:
    - Integer ports are converted to Port objects
    - Port objects can be provided directly
    - Custom ports override the default h2c port
    """
    # Test with integer port
    conn_int_port = ConnectorEnvironment(
        name="connector-int-port",
        image=Image.from_base("python:3.11"),
        port=9000,
    )
    assert isinstance(conn_int_port.port, Port)
    assert conn_int_port.port.port == 9000
    assert conn_int_port.port.name is None  # No name when using int

    # Test with Port object
    custom_port = Port(port=8888, name="custom-h2c")
    conn_port_obj = ConnectorEnvironment(
        name="connector-port-obj",
        image=Image.from_base("python:3.11"),
        port=custom_port,
    )
    assert conn_port_obj.port == custom_port
    assert conn_port_obj.port.port == 8888
    assert conn_port_obj.port.name == "custom-h2c"


def test_connector_environment_container_args_default():
    """
    GOAL: Verify that container_args returns connector-specific args when args is None.

    Tests that:
    - When args is None, returns ["c0", "--port", "<port>", "--prometheus_port", "9092"]
    - Port value is correctly extracted from port.port
    """
    conn_env = ConnectorEnvironment(
        name="connector-default-args",
        image=Image.from_base("python:3.11"),
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        root_dir=pathlib.Path.cwd(),
    )

    args = conn_env.container_args(ctx)
    assert args == ["c0", "--port", "8080", "--prometheus_port", "9092"]


def test_connector_environment_container_args_custom_port():
    """
    GOAL: Verify that container_args uses the custom port value.

    Tests that:
    - Custom port is reflected in the container_args
    - Port is converted to string in the args
    """
    conn_env = ConnectorEnvironment(
        name="connector-custom-port",
        image=Image.from_base("python:3.11"),
        port=9999,
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        root_dir=pathlib.Path.cwd(),
    )

    args = conn_env.container_args(ctx)
    assert args == ["c0", "--port", "9999", "--prometheus_port", "9092"]


def test_connector_environment_container_args_with_custom_args():
    """
    GOAL: Verify that container_args calls super() when custom args are provided.

    Tests that:
    - When args is provided (not None), super().container_args() is called
    - Custom args override the default connector args
    - Both list and string args work correctly
    """
    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        root_dir=pathlib.Path.cwd(),
    )

    # Test with list args
    conn_list_args = ConnectorEnvironment(
        name="connector-list-args",
        image=Image.from_base("python:3.11"),
        args=["--custom", "arg1", "--value", "arg2"],
    )
    assert conn_list_args.container_args(ctx) == ["--custom", "arg1", "--value", "arg2"]

    # Test with string args (will be split using shlex in parent class)
    conn_str_args = ConnectorEnvironment(
        name="connector-str-args",
        image=Image.from_base("python:3.11"),
        args="--host 0.0.0.0 --custom-port 3000",
    )
    assert conn_str_args.container_args(ctx) == ["--host", "0.0.0.0", "--custom-port", "3000"]


def test_connector_environment_container_cmd_string_command():
    """
    GOAL: Verify that container_cmd splits string commands using shlex.

    Tests that:
    - String commands are split into a list using shlex.split
    - Quoted strings in commands are preserved
    - The command is returned directly without additional processing
    """
    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        root_dir=pathlib.Path.cwd(),
    )

    # Test with simple string command
    conn_simple_cmd = ConnectorEnvironment(
        name="connector-simple-cmd",
        image=Image.from_base("python:3.11"),
        command="python app.py --debug",
    )
    cmd = conn_simple_cmd.container_cmd(ctx)
    assert cmd == ["python", "app.py", "--debug"]

    # Test with quoted string command
    conn_quoted_cmd = ConnectorEnvironment(
        name="connector-quoted-cmd",
        image=Image.from_base("python:3.11"),
        command='uvicorn app:main --host "0.0.0.0" --port 8000',
    )
    cmd_quoted = conn_quoted_cmd.container_cmd(ctx)
    assert cmd_quoted == ["uvicorn", "app:main", "--host", "0.0.0.0", "--port", "8000"]


def test_connector_environment_container_cmd_list_command():
    """
    GOAL: Verify that container_cmd returns list commands as-is.

    Tests that:
    - List commands are returned without modification
    - No shlex splitting occurs for list commands
    """
    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        root_dir=pathlib.Path.cwd(),
    )

    conn_list_cmd = ConnectorEnvironment(
        name="connector-list-cmd",
        image=Image.from_base("python:3.11"),
        command=["python", "-m", "connector.main", "--config", "config.yaml"],
    )
    cmd = conn_list_cmd.container_cmd(ctx)
    assert cmd == ["python", "-m", "connector.main", "--config", "config.yaml"]


def test_connector_environment_container_cmd_none_command():
    """
    GOAL: Verify that container_cmd uses default fserve command when command is None.

    Tests that:
    - When command is None, the parent class behavior is used (fserve command)
    - This is inherited from AppEnvironment
    """
    conn_env = ConnectorEnvironment(
        name="connector-no-cmd",
        image=Image.from_base("python:3.11"),
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=CodeBundle(computed_version="v1.0.0", tgz="s3://bucket/code.tgz"),
        root_dir=pathlib.Path.cwd(),
    )

    cmd = conn_env.container_cmd(ctx)
    # Should use default fserve command from parent class
    assert cmd[0] == "fserve"
    assert "--version" in cmd
    assert "v1.0.0" in cmd


def test_connector_environment_comprehensive_happy_path():
    """
    GOAL: Validate that ConnectorEnvironment works with all major features.

    Tests the complete lifecycle of creating a ConnectorEnvironment with:
    - All configuration parameters
    - Custom port
    - Custom args (using connector defaults)
    - Parameters
    - Container command and args generation

    This ensures all components work together harmoniously.
    """
    from flyte._resources import Resources
    from flyte.app._types import Domain, Link, Scaling

    conn_env = ConnectorEnvironment(
        name="my-connector-app",
        image=Image.from_base("python:3.11"),
        description="Test connector with all features",
        port=Port(port=8181, name="h2c"),
        command=None,  # Will use default fserve command
        args=None,  # Will use connector-specific args
        resources=Resources(cpu=2, memory="4Gi"),
        env_vars={"ENV": "production"},
        secrets="connector-secrets",
        requires_auth=True,
        scaling=Scaling(replicas=(1, 3), metric=Scaling.Concurrency(val=5)),
        domain=Domain(subdomain="connector"),
        links=[Link(path="/health", title="Health", is_relative=True)],
        parameters=[
            Parameter(value="connector.yaml", name="config", env_var="CONFIG_PATH"),
        ],
        cluster_pool="connector-pool",
    )

    # Verify basic properties
    assert conn_env.name == "my-connector-app"
    assert conn_env.type == "Connector"
    assert conn_env.port.port == 8181
    assert conn_env.port.name == "h2c"
    assert conn_env.description == "Test connector with all features"
    assert conn_env.cluster_pool == "connector-pool"

    # Test container_args with None args (connector defaults)
    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        root_dir=pathlib.Path.cwd(),
    )
    args = conn_env.container_args(ctx)
    assert args == ["c0", "--port", "8181", "--prometheus_port", "9092"]

    # Test container_cmd with code bundle
    ctx_with_bundle = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=CodeBundle(computed_version="v1.0.0", tgz="s3://bucket/code.tgz"),
        root_dir=pathlib.Path.cwd(),
    )
    cmd = conn_env.container_cmd(ctx_with_bundle)
    assert cmd[0] == "fserve"
    assert "--version" in cmd
    assert "--parameters" in cmd


def test_connector_environment_inherits_from_app_environment():
    """
    GOAL: Verify that ConnectorEnvironment properly inherits from AppEnvironment.

    Tests that:
    - ConnectorEnvironment is an instance of AppEnvironment
    - All AppEnvironment methods are available
    - Inheritance chain is correct
    """
    from flyte.app import AppEnvironment

    conn_env = ConnectorEnvironment(
        name="connector-inheritance",
        image=Image.from_base("python:3.11"),
    )

    # Verify inheritance
    assert isinstance(conn_env, AppEnvironment)
    assert isinstance(conn_env, ConnectorEnvironment)

    # Verify parent methods are available
    assert hasattr(conn_env, "container_args")
    assert hasattr(conn_env, "container_cmd")
    assert hasattr(conn_env, "_validate_name")
    assert hasattr(conn_env, "get_port")


def test_connector_environment_with_parameters():
    """
    GOAL: Verify that parameters work correctly with ConnectorEnvironment.

    Tests that:
    - Parameters can be configured
    - Parameters are serialized in container_cmd
    - Both string and File/Dir parameters work
    """
    conn_env = ConnectorEnvironment(
        name="connector-params",
        image=Image.from_base("python:3.11"),
        parameters=[
            Parameter(value="config.yaml", name="config", env_var="CONFIG_FILE"),
            Parameter(value="s3://bucket/data", name="data", download=True),
        ],
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=CodeBundle(computed_version="v1.0.0", tgz="s3://bucket/code.tgz"),
        root_dir=pathlib.Path.cwd(),
    )

    cmd = conn_env.container_cmd(ctx)
    assert "--parameters" in cmd

    # Verify parameters can be deserialized
    from flyte.app._parameter import SerializableParameterCollection

    parameters_idx = cmd.index("--parameters")
    serialized = cmd[parameters_idx + 1]
    deserialized = SerializableParameterCollection.from_transport(serialized)
    assert len(deserialized.parameters) == 2
    assert deserialized.parameters[0].name == "config"
    assert deserialized.parameters[1].name == "data"


def test_connector_environment_invalid_port():
    """
    GOAL: Verify that invalid ports are rejected.

    Tests that:
    - Reserved ports (8012, 8022, 8112, 9090, 9091) raise ValueError
    - This validation is inherited from AppEnvironment
    """
    invalid_ports = [8012, 8022, 8112, 9090, 9091]
    for invalid_port in invalid_ports:
        with pytest.raises(ValueError, match="is not allowed"):
            ConnectorEnvironment(
                name="connector-invalid-port",
                image=Image.from_base("python:3.11"),
                port=invalid_port,
            )


def test_connector_environment_name_validation():
    """
    GOAL: Verify that connector names are validated according to Kubernetes naming rules.

    Tests that:
    - Valid names (lowercase alphanumeric with hyphens) are accepted
    - Invalid names raise ValueError
    - Name validation is inherited from AppEnvironment
    """
    # Valid names should work
    valid_names = ["my-connector", "connector123", "c0", "my-connector-app"]
    for name in valid_names:
        conn = ConnectorEnvironment(
            name=name,
            image=Image.from_base("python:3.11"),
        )
        assert conn.name == name
        conn._validate_name()  # Should not raise

    # Invalid names should fail validation
    invalid_names = ["My-Connector", "my_connector", "-connector", "connector-"]
    for name in invalid_names:
        with pytest.raises(ValueError, match="must consist of lower case"):
            ConnectorEnvironment(
                name=name,
                image=Image.from_base("python:3.11"),
            )


def test_connector_environment_with_lifecycle_decorators():
    """
    GOAL: Verify that lifecycle decorators work with ConnectorEnvironment.

    Tests that:
    - server, on_startup, and on_shutdown decorators can be used
    - These are inherited from AppEnvironment
    """
    conn = ConnectorEnvironment(
        name="connector-lifecycle",
        image=Image.from_base("python:3.11"),
    )

    @conn.on_startup
    def startup():
        """Startup function."""

    @conn.server
    def server():
        """Server function."""

    @conn.on_shutdown
    def shutdown():
        """Shutdown function."""

    assert conn._on_startup is not None
    assert conn._on_startup == startup
    assert conn._server is not None
    assert conn._server == server
    assert conn._on_shutdown is not None
    assert conn._on_shutdown == shutdown


def test_connector_environment_container_cmd_with_parameter_overrides():
    """
    GOAL: Verify that parameter_overrides work correctly with ConnectorEnvironment.

    Tests that:
    - parameter_overrides can be passed to container_cmd
    - Overridden values are used in serialization
    - This functionality is inherited from AppEnvironment
    """
    from dataclasses import replace

    from flyte.app._parameter import SerializableParameterCollection

    conn_env = ConnectorEnvironment(
        name="connector-param-override",
        image=Image.from_base("python:3.11"),
        parameters=[
            Parameter(value="original.yaml", name="config"),
        ],
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=CodeBundle(computed_version="v1.0.0", tgz="s3://bucket/code.tgz"),
        root_dir=pathlib.Path.cwd(),
    )

    # Create override
    parameter_overrides = [
        replace(conn_env.parameters[0], value="overridden.yaml"),
    ]

    cmd = conn_env.container_cmd(ctx, parameter_overrides=parameter_overrides)
    parameters_idx = cmd.index("--parameters")
    serialized = cmd[parameters_idx + 1]
    deserialized = SerializableParameterCollection.from_transport(serialized)

    assert deserialized.parameters[0].value == "overridden.yaml"


def test_connector_environment_empty_args_vs_none_args():
    """
    GOAL: Verify the difference between args=None and args=[].

    Tests that:
    - args=None triggers connector-specific default args
    - args=[] (empty list) returns empty list (not connector defaults)
    - This demonstrates the None check in container_args
    """
    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        root_dir=pathlib.Path.cwd(),
    )

    # None args should use connector defaults
    conn_none = ConnectorEnvironment(
        name="connector-none-args",
        image=Image.from_base("python:3.11"),
        args=None,
    )
    assert conn_none.container_args(ctx) == ["c0", "--port", "8080", "--prometheus_port", "9092"]

    # Empty list should return empty list (not connector defaults)
    conn_empty = ConnectorEnvironment(
        name="connector-empty-args",
        image=Image.from_base("python:3.11"),
        args=[],
    )
    assert conn_empty.container_args(ctx) == []
