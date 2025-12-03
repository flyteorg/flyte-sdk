"""
Comprehensive unit tests for AppEnvironment.

These tests verify AppEnvironment functionality without using mocks,
focusing on container_cmd, container_args, and Input handling.
"""

import pytest

from flyte._image import Image
from flyte._internal.imagebuild.image_builder import ImageCache
from flyte._resources import Resources
from flyte.app import AppEnvironment
from flyte.app._input import Input
from flyte.app._types import Domain, Link, Port, Scaling
from flyte.models import CodeBundle, SerializationContext

# Import flyte.io first and inject into app._input module to fix NameError bug


def test_app_environment_comprehensive_happy_path():
    """
    GOAL: Validate that AppEnvironment correctly handles all major features together.

    Tests the complete lifecycle of creating an AppEnvironment with:
    - All configuration parameters (image, resources, env_vars, secrets, scaling, domain, links)
    - Port conversion from int to Port object
    - Input serialization and inclusion in container_cmd
    - Command generation with code bundle, version, and inputs
    - Args handling

    This ensures all components work together harmoniously.
    """
    # Create comprehensive AppEnvironment with all features
    app_env = AppEnvironment(
        name="my-test-app",
        image=Image.from_base("python:3.11"),
        type="streamlit",
        description="Test application with all features",
        port=8501,  # Will be converted to Port
        command=None,  # Will use default command
        args=["--arg1", "value1"],
        resources=Resources(cpu=2, memory="4Gi", gpu=1),
        env_vars={"ENV": "production", "DEBUG": "false"},
        secrets="my-secret-group",
        requires_auth=True,
        scaling=Scaling(replicas=(2, 5), metric=Scaling.Concurrency(val=10)),
        domain=Domain(subdomain="myapp", custom_domain="example.com"),
        links=[
            Link(path="/health", title="Health Check", is_relative=True),
            Link(path="/docs", title="Documentation", is_relative=True),
        ],
        inputs=[
            Input(value="config.yaml", name="config", env_var="CONFIG_PATH"),
            Input(value="s3://bucket/data", name="data", download=True, mount="/mnt/data"),
        ],
        cluster_pool="gpu-pool",
        include=["*.py", "requirements.txt"],
    )

    # Verify basic properties are correctly set
    assert app_env.name == "my-test-app"
    assert app_env.type == "streamlit"
    assert app_env.description == "Test application with all features"
    assert app_env.cluster_pool == "gpu-pool"
    assert app_env.requires_auth is True

    # Verify port was converted from int to Port object
    assert isinstance(app_env.port, Port)
    assert app_env.get_port().port == 8501
    assert app_env.get_port().name is None

    # Verify resources are correctly set
    assert app_env.resources.cpu == 2
    assert app_env.resources.memory == "4Gi"
    assert app_env.resources.gpu == 1

    # Verify environment variables
    assert app_env.env_vars == {"ENV": "production", "DEBUG": "false"}

    # Verify secrets
    assert app_env.secrets == "my-secret-group"

    # Verify scaling configuration
    assert app_env.scaling.replicas == (2, 5)
    assert isinstance(app_env.scaling.metric, Scaling.Concurrency)
    assert app_env.scaling.metric.val == 10

    # Verify domain configuration
    assert app_env.domain.subdomain == "myapp"
    assert app_env.domain.custom_domain == "example.com"

    # Verify links are correctly set
    assert len(app_env.links) == 2
    assert app_env.links[0].path == "/health"
    assert app_env.links[0].title == "Health Check"
    assert app_env.links[1].path == "/docs"

    # Verify inputs are correctly set
    assert len(app_env.inputs) == 2
    assert app_env.inputs[0].name == "config"
    assert app_env.inputs[0].value == "config.yaml"
    assert app_env.inputs[0].env_var == "CONFIG_PATH"
    assert app_env.inputs[1].name == "data"
    assert app_env.inputs[1].download is True
    assert app_env.inputs[1].mount == "/mnt/data"

    # Verify includes
    assert app_env.include == ["*.py", "requirements.txt"]

    # Test container_args returns the args as-is (list format)
    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
    )
    args = app_env.container_args(ctx)
    assert args == ["--arg1", "value1"]

    # Test container_cmd generates correct command with inputs serialized
    ctx_with_bundle = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=CodeBundle(computed_version="v1.0.0", tgz="s3://bucket/code.tgz", destination="/app"),
    )
    cmd = app_env.container_cmd(ctx_with_bundle)

    # Verify command structure includes all required flags
    assert cmd[0] == "fserve"
    assert "--version" in cmd
    assert "v1.0.0" in cmd
    assert "--project" in cmd
    assert "test-project" in cmd
    assert "--domain" in cmd
    assert "test-domain" in cmd
    assert "--org" in cmd
    assert "test-org" in cmd
    assert "--tgz" in cmd
    assert "s3://bucket/code.tgz" in cmd
    assert "--dest" in cmd
    assert "/app" in cmd
    assert "--inputs" in cmd
    # Inputs should be serialized (base64 gzip encoded)
    inputs_idx = cmd.index("--inputs")
    assert inputs_idx >= 0
    assert len(cmd[inputs_idx + 1]) > 0  # Should have serialized inputs
    assert cmd[-1] == "--"  # Command should end with "--"


def test_app_environment_container_cmd_with_inputs():
    """
    GOAL: Verify that inputs are correctly serialized and included in container_cmd.

    Tests that:
    - Multiple inputs can be configured
    - Each input with different properties (env_var, download) is handled
    - Inputs are serialized into base64 gzip format
    - Serialized inputs can be deserialized back to verify correctness
    """
    app_env = AppEnvironment(
        name="app-with-inputs",
        image=Image.from_base("python:3.11"),
        inputs=[
            Input(value="file1.txt", name="input1", env_var="INPUT1"),
            Input(value="file2.txt", name="input2"),
            Input(value="s3://bucket/file3.txt", name="input3", download=True),
        ],
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=CodeBundle(computed_version="v1.0.0", tgz="s3://bucket/code.tgz"),
    )

    cmd = app_env.container_cmd(ctx)

    # Verify --inputs flag is present
    assert "--inputs" in cmd
    inputs_idx = cmd.index("--inputs")
    serialized_inputs = cmd[inputs_idx + 1]

    # Verify serialized inputs can be deserialized correctly
    from flyte.app._input import SerializableInputCollection

    deserialized = SerializableInputCollection.from_transport(serialized_inputs)
    assert len(deserialized.inputs) == 3
    assert deserialized.inputs[0].name == "input1"
    assert deserialized.inputs[0].value == "file1.txt"
    assert deserialized.inputs[0].env_var == "INPUT1"
    assert deserialized.inputs[1].name == "input2"
    assert deserialized.inputs[2].name == "input3"
    assert deserialized.inputs[2].download is False  # String type doesn't auto-download


def test_app_environment_container_cmd_without_inputs():
    """
    GOAL: Verify that container_cmd works correctly when no inputs are provided.

    Tests that the --inputs flag is NOT added to the command when inputs list is empty,
    ensuring the command is clean and doesn't include unnecessary flags.
    """
    app_env = AppEnvironment(
        name="app-no-inputs",
        image=Image.from_base("python:3.11"),
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=CodeBundle(computed_version="v1.0.0", tgz="s3://bucket/code.tgz"),
    )

    cmd = app_env.container_cmd(ctx)

    # Verify inputs flag is NOT in command when no inputs
    assert "--inputs" not in cmd
    assert cmd[-1] == "--"


def test_app_environment_container_cmd_custom_command():
    """
    GOAL: Verify that custom commands override the default fserve command.

    Tests that:
    - List-format custom commands are used as-is
    - String-format custom commands are split using shlex
    - Custom commands completely replace the default fserve command
    - Inputs are NOT added when using custom commands (they're user-managed)
    """
    # Test with list command
    app_env_list = AppEnvironment(
        name="app-custom-cmd-list",
        image=Image.from_base("python:3.11"),
        command=["python", "app.py"],
        inputs=[Input(value="config.yaml")],  # Inputs should be ignored with custom command
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
    )

    cmd_list = app_env_list.container_cmd(ctx)
    assert cmd_list == ["python", "app.py"]
    assert "--inputs" not in cmd_list  # Inputs not added for custom commands

    # Test with string command (will be split using shlex)
    app_env_str = AppEnvironment(
        name="app-custom-cmd-str",
        image=Image.from_base("python:3.11"),
        command="uvicorn app:main --host 0.0.0.0",
    )

    cmd_str = app_env_str.container_cmd(ctx)
    assert cmd_str == ["uvicorn", "app:main", "--host", "0.0.0.0"]


def test_app_environment_container_args_variations():
    """
    GOAL: Verify that container_args correctly handles different arg formats.

    Tests that:
    - List args are returned as-is
    - String args are split using shlex
    - None args return empty list
    - Quoted string args preserve quoted content
    """
    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
    )

    # Test with list args
    app_list = AppEnvironment(
        name="app-list-args",
        image=Image.from_base("python:3.11"),
        args=["--arg1", "value1", "--arg2", "value2"],
    )
    assert app_list.container_args(ctx) == ["--arg1", "value1", "--arg2", "value2"]

    # Test with string args (will be split using shlex)
    app_str = AppEnvironment(
        name="app-str-args",
        image=Image.from_base("python:3.11"),
        args="--host 0.0.0.0 --port 8080",
    )
    assert app_str.container_args(ctx) == ["--host", "0.0.0.0", "--port", "8080"]

    # Test with None args
    app_none = AppEnvironment(
        name="app-none-args",
        image=Image.from_base("python:3.11"),
        args=None,
    )
    assert app_none.container_args(ctx) == []

    # Test with quoted string args (shlex preserves quoted content)
    app_quoted = AppEnvironment(
        name="app-quoted-args",
        image=Image.from_base("python:3.11"),
        args='--message "Hello World" --count 5',
    )
    assert app_quoted.container_args(ctx) == ["--message", "Hello World", "--count", "5"]


def test_app_environment_container_cmd_with_image_cache():
    """
    GOAL: Verify that image cache is correctly included in container_cmd.

    Tests that:
    - Serialized image cache form is used when available
    - Non-serialized image cache falls back to to_transport
    - --image-cache flag is properly added to the command
    """
    app_env = AppEnvironment(
        name="app-with-cache",
        image=Image.from_base("python:3.11"),
    )

    # Test with serialized image cache (uses serialized_form directly)
    ctx_serialized = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=CodeBundle(computed_version="v1.0.0", tgz="s3://bucket/code.tgz"),
        image_cache=ImageCache(image_lookup={"default": "python:3.11"}, serialized_form="base64encodedcache"),
    )

    cmd = app_env.container_cmd(ctx_serialized)
    assert "--image-cache" in cmd
    cache_idx = cmd.index("--image-cache")
    assert cmd[cache_idx + 1] == "base64encodedcache"

    # Test with non-serialized image cache (calls to_transport)
    ctx_non_serialized = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=CodeBundle(computed_version="v1.0.0", tgz="s3://bucket/code.tgz"),
        image_cache=ImageCache(image_lookup={"default": "python:3.11"}),
    )

    cmd2 = app_env.container_cmd(ctx_non_serialized)
    assert "--image-cache" in cmd2


def test_app_environment_container_cmd_with_pkl_bundle():
    """
    GOAL: Verify that pkl code bundles work as an alternative to tgz bundles.

    Tests that:
    - --pkl flag is used instead of --tgz when pkl bundle is provided
    - --dest flag is still included for destination directory
    """
    app_env = AppEnvironment(
        name="app-with-pkl",
        image=Image.from_base("python:3.11"),
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=CodeBundle(computed_version="v1.0.0", pkl="s3://bucket/code.pkl", destination="/app"),
    )

    cmd = app_env.container_cmd(ctx)
    assert "--pkl" in cmd
    assert "s3://bucket/code.pkl" in cmd
    assert "--tgz" not in cmd  # tgz should not be present when using pkl


def test_app_environment_port_handling():
    """
    GOAL: Verify correct port handling and validation.

    Tests that:
    - Integer ports are automatically converted to Port objects
    - Port objects are preserved as-is
    - Reserved ports (8012, 8022, 8112, 9090, 9091) raise ValueError
    - get_port() returns the correct Port object
    """
    # Test with integer port (auto-converted to Port)
    app_int_port = AppEnvironment(
        name="app-int-port",
        image=Image.from_base("python:3.11"),
        port=9000,
    )
    assert isinstance(app_int_port.port, Port)
    assert app_int_port.port.port == 9000
    assert app_int_port.port.name is None
    assert app_int_port.get_port().port == 9000

    # Test with Port object (preserved as-is)
    custom_port = Port(port=8080, name="custom")
    app_port_obj = AppEnvironment(
        name="app-port-obj",
        image=Image.from_base("python:3.11"),
        port=custom_port,
    )
    assert app_port_obj.port == custom_port
    assert app_port_obj.get_port().port == 8080
    assert app_port_obj.get_port().name == "custom"

    # Test invalid ports raise ValueError
    invalid_ports = [8012, 8022, 8112, 9090, 9091]
    for invalid_port in invalid_ports:
        with pytest.raises(ValueError, match="is not allowed"):
            AppEnvironment(
                name="app-invalid-port",
                image=Image.from_base("python:3.11"),
                port=invalid_port,
            )


def test_app_environment_name_validation():
    """
    GOAL: Verify that app names are validated according to Kubernetes naming rules.

    Tests that:
    - Valid names (lowercase alphanumeric with hyphens) are accepted
    - Invalid names fail _validate_name() but not __init__ (validation is explicit)
    - Name validation follows the regex: [a-z0-9]([-a-z0-9]*[a-z0-9])?
    """
    # Valid names should work
    valid_names = ["myapp", "my-app", "app123", "my-app-123"]
    for name in valid_names:
        app = AppEnvironment(
            name=name,
            image=Image.from_base("python:3.11"),
        )
        assert app.name == name
        app._validate_name()  # Should not raise

    # Invalid names should fail validation
    invalid_names = ["My-App", "my_app", "-myapp", "myapp-", "my..app"]
    for name in invalid_names:
        with pytest.raises(ValueError, match="must consist of lower case"):
            AppEnvironment(
                name=name,
                image=Image.from_base("python:3.11"),
            )


def test_app_environment_type_validation():
    """
    GOAL: Verify that type validation catches incorrect parameter types.

    Tests __post_init__ validation for:
    - args must be List[str] or str
    - command must be List[str] or str
    - scaling must be Scaling type
    - domain must be Domain or None
    - links must be List[Link]
    """
    # Invalid args type
    with pytest.raises(TypeError, match="Expected args to be of type List\\[str\\] or str"):
        AppEnvironment(
            name="test-app",
            image=Image.from_base("python:3.11"),
            args=123,  # type: ignore
        )

    # Invalid command type
    with pytest.raises(TypeError, match="Expected command to be of type List\\[str\\] or str"):
        AppEnvironment(
            name="test-app",
            image=Image.from_base("python:3.11"),
            command={"key": "value"},  # type: ignore
        )

    # Invalid scaling type
    with pytest.raises(TypeError, match="Expected scaling to be of type Scaling"):
        AppEnvironment(
            name="test-app",
            image=Image.from_base("python:3.11"),
            scaling="invalid",  # type: ignore
        )

    # Invalid domain type
    with pytest.raises(TypeError, match="Expected domain to be of type Domain or None"):
        AppEnvironment(
            name="test-app",
            image=Image.from_base("python:3.11"),
            domain="invalid",  # type: ignore
        )

    # Invalid links type
    with pytest.raises(TypeError, match="Expected links to be of type List\\[Link\\]"):
        AppEnvironment(
            name="test-app",
            image=Image.from_base("python:3.11"),
            links=["invalid"],  # type: ignore
        )


def test_app_environment_scaling_configurations():
    """
    GOAL: Verify various scaling configurations work correctly.

    Tests that:
    - Single replica value creates (N, N) tuple
    - Min/max replica tuple is preserved
    - Concurrency metric is correctly set
    - RequestRate metric is correctly set
    """
    # Basic scaling with single replica value
    app_basic = AppEnvironment(
        name="app-basic-scaling",
        image=Image.from_base("python:3.11"),
        scaling=Scaling(replicas=3),
    )
    assert app_basic.scaling.replicas == (3, 3)

    # Scaling with min/max replicas tuple
    app_minmax = AppEnvironment(
        name="app-minmax-scaling",
        image=Image.from_base("python:3.11"),
        scaling=Scaling(replicas=(1, 10)),
    )
    assert app_minmax.scaling.replicas == (1, 10)

    # Scaling with concurrency metric
    app_concurrency = AppEnvironment(
        name="app-concurrency",
        image=Image.from_base("python:3.11"),
        scaling=Scaling(replicas=(2, 5), metric=Scaling.Concurrency(val=20)),
    )
    assert isinstance(app_concurrency.scaling.metric, Scaling.Concurrency)
    assert app_concurrency.scaling.metric.val == 20

    # Scaling with request rate metric
    app_rate = AppEnvironment(
        name="app-rate",
        image=Image.from_base("python:3.11"),
        scaling=Scaling(replicas=(1, 8), metric=Scaling.RequestRate(val=100)),
    )
    assert isinstance(app_rate.scaling.metric, Scaling.RequestRate)
    assert app_rate.scaling.metric.val == 100


def test_app_environment_default_values():
    """
    GOAL: Verify that all default values are correctly set for minimal AppEnvironment.

    Tests that a minimal AppEnvironment (only name and image) has sensible defaults:
    - Port defaults to 8080
    - No custom command/args
    - Auth required by default
    - Single replica scaling
    - Default cluster pool
    - Empty lists for optional collections
    """
    app = AppEnvironment(
        name="minimal-app",
        image=Image.from_base("python:3.11"),
    )

    # Check all defaults
    assert app.type is None
    assert app.description is None
    assert isinstance(app.port, Port)
    assert app.port.port == 8080
    assert app.args is None
    assert app.command is None
    assert app.requires_auth is True
    assert isinstance(app.scaling, Scaling)
    assert app.scaling.replicas == (0, 1)
    assert isinstance(app.domain, Domain)
    assert app.links == []
    assert app.inputs == []
    assert app.cluster_pool == "default"
    assert app.include == []
    assert app.env_vars is None
    assert app.secrets is None


def test_app_environment_with_file_and_dir_inputs():
    """
    GOAL: Verify that File and Dir inputs are correctly serialized.

    Tests that:
    - File inputs are serialized with type="file"
    - Dir inputs are serialized with type="directory"
    - String inputs are serialized with type="string"
    - Mount paths enable auto-download
    - Ignore patterns are preserved for directories
    """
    from flyte.io import Dir, File

    # Create File and Dir objects
    file_input = File(path="s3://bucket/file.txt")
    dir_input = Dir(path="s3://bucket/directory")

    app_env = AppEnvironment(
        name="app-with-file-dir",
        image=Image.from_base("python:3.11"),
        inputs=[
            Input(value=file_input, name="myfile", mount="/mnt/file"),
            Input(value=dir_input, name="mydir", mount="/mnt/dir", ignore_patterns=["*.log", "*.tmp"]),
            Input(value="plain-string", name="mystring"),
        ],
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=CodeBundle(computed_version="v1.0.0", tgz="s3://bucket/code.tgz"),
    )

    cmd = app_env.container_cmd(ctx)
    assert "--inputs" in cmd

    # Deserialize and verify types and properties
    inputs_idx = cmd.index("--inputs")
    serialized = cmd[inputs_idx + 1]

    from flyte.app._input import SerializableInputCollection

    deserialized = SerializableInputCollection.from_transport(serialized)
    assert len(deserialized.inputs) == 3

    # File input
    assert deserialized.inputs[0].name == "myfile"
    assert deserialized.inputs[0].type == "file"
    assert deserialized.inputs[0].value == "s3://bucket/file.txt"
    assert deserialized.inputs[0].download is True  # mount implies download
    assert deserialized.inputs[0].dest == "/mnt/file"

    # Dir input
    assert deserialized.inputs[1].name == "mydir"
    assert deserialized.inputs[1].type == "directory"
    assert deserialized.inputs[1].value == "s3://bucket/directory"
    assert deserialized.inputs[1].download is True
    assert deserialized.inputs[1].dest == "/mnt/dir"
    assert deserialized.inputs[1].ignore_patterns == ["*.log", "*.tmp"]

    # String input
    assert deserialized.inputs[2].name == "mystring"
    assert deserialized.inputs[2].type == "string"
    assert deserialized.inputs[2].value == "plain-string"
    assert deserialized.inputs[2].download is False


def test_app_environment_empty_inputs():
    """
    GOAL: Verify that empty inputs list doesn't add unnecessary --inputs flag.

    Tests that when inputs=[] (empty list), the container_cmd doesn't include
    the --inputs flag, keeping the command clean.
    """
    app_env = AppEnvironment(
        name="app-empty-inputs",
        image=Image.from_base("python:3.11"),
        inputs=[],
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=CodeBundle(computed_version="v1.0.0", tgz="s3://bucket/code.tgz"),
    )

    cmd = app_env.container_cmd(ctx)
    assert "--inputs" not in cmd


def test_app_environment_container_cmd_version_handling():
    """
    GOAL: Verify that version precedence is correct in container_cmd.

    Tests that:
    - Explicit version in SerializationContext takes precedence
    - Falls back to computed_version when version is None
    """
    app_env = AppEnvironment(
        name="app-version",
        image=Image.from_base("python:3.11"),
    )

    # Test with explicit version (takes precedence)
    ctx_explicit = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v2.5.0",
        code_bundle=CodeBundle(computed_version="v2.5.1", tgz="s3://bucket/code.tgz"),
    )

    cmd = app_env.container_cmd(ctx_explicit)
    assert "--version" in cmd
    ver_idx = cmd.index("--version")
    assert cmd[ver_idx + 1] == "v2.5.0"  # Explicit version takes precedence

    # Test with computed version (fallback when version is None)
    ctx_computed = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version=None,  # type: ignore
        code_bundle=CodeBundle(computed_version="v2.5.1", tgz="s3://bucket/code.tgz"),
    )

    cmd2 = app_env.container_cmd(ctx_computed)
    ver_idx2 = cmd2.index("--version")
    assert cmd2[ver_idx2 + 1] == "v2.5.1"  # Falls back to computed version


def test_app_environment_container_cmd_no_code_bundle():
    """
    GOAL: Verify that container_cmd works when code_bundle is None.

    Tests that the command is still generated with fserve flags but without
    code bundle related flags (--tgz, --pkl, --dest).
    """
    app_env = AppEnvironment(
        name="app-no-bundle",
        image=Image.from_base("python:3.11"),
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=None,
    )

    cmd = app_env.container_cmd(ctx)
    # Should still generate command but without code bundle flags
    assert cmd[0] == "fserve"
    assert "--version" in cmd
    assert "--tgz" not in cmd
    assert "--pkl" not in cmd
    assert "--dest" not in cmd


def test_app_environment_multiple_links():
    """
    GOAL: Verify that multiple links can be configured correctly.

    Tests that:
    - Multiple Link objects can be added
    - Both relative and absolute links are supported
    - All link properties (path, title, is_relative) are preserved
    """
    app_env = AppEnvironment(
        name="app-links",
        image=Image.from_base("python:3.11"),
        links=[
            Link(path="/", title="Home", is_relative=True),
            Link(path="/api/docs", title="API Docs", is_relative=True),
            Link(path="https://external.com", title="External", is_relative=False),
            Link(path="/metrics", title="Metrics", is_relative=True),
        ],
    )

    assert len(app_env.links) == 4
    assert all(isinstance(link, Link) for link in app_env.links)
    assert app_env.links[0].path == "/"
    assert app_env.links[2].is_relative is False
    assert app_env.links[2].path == "https://external.com"
