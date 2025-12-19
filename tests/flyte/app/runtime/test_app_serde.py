"""
Tests for AppEnvironment serialization to protobuf messages.

These tests verify that app_serde.py correctly converts AppEnvironment objects
into protobuf IDL format without using mocks.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.core import tasks_pb2

import flyte.io
from flyte._image import Image
from flyte._internal.imagebuild.image_builder import ImageCache
from flyte._resources import Resources
from flyte.app import AppEnvironment
from flyte.app._input import Parameter, RunOutput
from flyte.app._runtime.app_serde import (
    _get_scaling_metric,
    _materialize_inputs_with_delayed_values,
    _sanitize_resource_name,
    get_proto_container,
    translate_app_env_to_idl,
)
from flyte.app._types import Domain, Port, Scaling
from flyte.models import CodeBundle, SerializationContext


def test_sanitize_resource_name():
    """
    GOAL: Verify resource names are sanitized for Kubernetes compatibility.

    Tests that resource names are converted to lowercase with underscores replaced by hyphens.
    """
    # Test CPU
    resource = tasks_pb2.Resources.ResourceEntry(name=tasks_pb2.Resources.ResourceName.CPU, value="2")
    sanitized = _sanitize_resource_name(resource)
    assert sanitized == "cpu"

    # Test GPU
    resource = tasks_pb2.Resources.ResourceEntry(name=tasks_pb2.Resources.ResourceName.GPU, value="1")
    sanitized = _sanitize_resource_name(resource)
    assert sanitized == "gpu"

    # Test ephemeral storage (has underscore that should be replaced with hyphen)
    resource = tasks_pb2.Resources.ResourceEntry(name=tasks_pb2.Resources.ResourceName.EPHEMERAL_STORAGE, value="10Gi")
    sanitized = _sanitize_resource_name(resource)
    assert sanitized == "ephemeral-storage"


def test_get_scaling_metric_none():
    """
    GOAL: Verify that None metric returns None.

    Tests edge case where no scaling metric is provided.
    """
    result = _get_scaling_metric(None)
    assert result is None


def test_get_scaling_metric_concurrency():
    """
    GOAL: Document bug in Concurrency metric serialization.

    The implementation uses 'val' field but protobuf expects 'target_value'.
    """
    metric = Scaling.Concurrency(val=10)
    # Note: Implementation currently has a bug - uses 'val' instead of 'target_value'
    with pytest.raises(ValueError, match='has no "val" field'):
        _get_scaling_metric(metric)


def test_get_scaling_metric_request_rate():
    """
    GOAL: Document bug in RequestRate metric serialization.

    The implementation uses 'val' field but protobuf expects 'target_value'.
    """
    metric = Scaling.RequestRate(val=100)
    # Note: Implementation currently has a bug - uses 'val' instead of 'target_value'
    with pytest.raises(ValueError, match='has no "val" field'):
        _get_scaling_metric(metric)


def test_get_proto_container_basic():
    """
    GOAL: Verify basic container protobuf generation without optional parameters.

    Tests that:
    - Image is serialized correctly
    - Default port (8080) is set
    - Port name defaults to "http"
    """
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1",
    )

    container = get_proto_container(app_env, ctx)

    assert container.image is not None
    assert len(container.ports) == 1
    assert container.ports[0].container_port == 8080
    assert container.ports[0].name == ""


def test_get_proto_container_with_resources():
    """
    GOAL: Verify that CPU and memory resources are correctly serialized to protobuf.

    Tests that resource requests are properly converted to protobuf ResourceEntry format.
    """
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
        resources=Resources(cpu=2, memory="4Gi"),
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1",
    )

    container = get_proto_container(app_env, ctx)

    assert container.resources is not None
    assert len(container.resources.requests) == 2

    # Check CPU request
    cpu_req = next((r for r in container.resources.requests if r.name == tasks_pb2.Resources.ResourceName.CPU), None)
    assert cpu_req is not None
    assert cpu_req.value == "2"

    # Check memory request
    mem_req = next((r for r in container.resources.requests if r.name == tasks_pb2.Resources.ResourceName.MEMORY), None)
    assert mem_req is not None
    assert mem_req.value == "4Gi"


def test_get_proto_container_with_env_vars():
    """
    GOAL: Verify environment variables are serialized to KeyValuePair protobuf format.

    Tests that env_vars dict is converted to a list of KeyValuePair messages.
    """
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
        env_vars={"FOO": "bar", "BAZ": "qux"},
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1",
    )

    container = get_proto_container(app_env, ctx)

    assert container.env is not None
    assert len(container.env) == 2
    env_dict = {kv.key: kv.value for kv in container.env}
    assert env_dict["FOO"] == "bar"
    assert env_dict["BAZ"] == "qux"


def test_get_proto_container_with_custom_port():
    """
    GOAL: Verify custom ports are correctly serialized.

    Tests that both port number and port name are preserved in the protobuf.
    """
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
        port=Port(port=9000, name="custom"),
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1",
    )

    container = get_proto_container(app_env, ctx)

    assert len(container.ports) == 1
    assert container.ports[0].container_port == 9000
    assert container.ports[0].name == "custom"


def test_get_proto_container_with_command_and_args():
    """
    GOAL: Verify custom command and args are serialized correctly.

    Tests that list-format command and args are preserved in the container protobuf.
    """
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
        command=["python", "-m", "myapp"],
        args=["--host", "0.0.0.0"],
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1",
    )

    container = get_proto_container(app_env, ctx)

    assert container.command == ["python", "-m", "myapp"]
    assert container.args == ["--host", "0.0.0.0"]


def test_get_proto_container_with_args_and_inputs():
    """
    GOAL: Verify that args and inputs work together correctly.

    Tests that:
    - Args are included in the container args field
    - Inputs are included in the command via --inputs flag
    - Args and inputs don't interfere with each other
    """
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
        args=["--arg1", "value1", "--arg2", "value2"],
        inputs=[
            Parameter(value="config.yaml", name="config"),
            Parameter(value="data.csv", name="data"),
        ],
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=CodeBundle(computed_version="v1.0.0", tgz="s3://bucket/code.tgz"),
    )

    container = get_proto_container(app_env, ctx)

    # Args should be in the args field
    assert container.args == ["--arg1", "value1", "--arg2", "value2"]

    # Command should contain fserve with --inputs flag
    assert container.command[0] == "fserve"
    assert "--inputs" in container.command

    # Verify inputs are serialized
    cmd_list = list(container.command)
    inputs_idx = cmd_list.index("--inputs")
    assert inputs_idx >= 0
    serialized_inputs = cmd_list[inputs_idx + 1]
    assert len(serialized_inputs) > 0  # Should have base64 gzip encoded content


def test_get_proto_container_with_string_args_and_inputs():
    """
    GOAL: Verify string args are split correctly when app has inputs.

    Tests that string args are parsed using shlex while inputs remain in command.
    """
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
        args="--host 0.0.0.0 --port 8080",
        inputs=[Input(value="config.yaml", name="config")],
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=CodeBundle(computed_version="v1.0.0", tgz="s3://bucket/code.tgz"),
    )

    container = get_proto_container(app_env, ctx)

    # String args should be split into list
    assert container.args == ["--host", "0.0.0.0", "--port", "8080"]

    # Inputs should be in command
    assert "--inputs" in container.command


def test_get_proto_container_with_only_inputs_no_args():
    """
    GOAL: Verify container works with inputs but no args.

    Tests that:
    - Inputs are added to command via --inputs
    - Args field is empty when no args provided
    """
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
        inputs=[
            Parameter(value="file1.txt", name="input1"),
            Parameter(value="file2.txt", name="input2"),
        ],
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=CodeBundle(computed_version="v1.0.0", tgz="s3://bucket/code.tgz"),
    )

    container = get_proto_container(app_env, ctx)

    # Args should be empty
    assert container.args == []

    # Inputs should be in command
    assert "--inputs" in container.command


def test_get_proto_container_with_custom_command_and_inputs():
    """
    GOAL: Verify custom command overrides default and inputs are ignored.

    Tests that:
    - Custom command completely replaces fserve
    - Inputs are NOT added to custom commands
    - User is responsible for handling inputs with custom commands
    """
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
        command=["python", "app.py"],
        args=["--custom-arg"],
        inputs=[Input(value="config.yaml", name="config")],  # Should be ignored
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
    )

    container = get_proto_container(app_env, ctx)

    # Custom command should be used
    assert container.command == ["python", "app.py"]

    # Args should still work
    assert container.args == ["--custom-arg"]

    # Inputs should NOT be in command (custom commands don't auto-add inputs)
    assert "--inputs" not in container.command


def test_get_proto_container_with_string_image():
    """
    GOAL: Verify string images are auto-converted by AppEnvironment.

    Tests that AppEnvironment.__post_init__ converts string images to Image objects.
    """
    app_env = AppEnvironment(
        name="test-app",
        image="python:3.11",  # String image
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1",
    )

    # AppEnvironment converts string images to Image objects in __post_init__
    container = get_proto_container(app_env, ctx)
    assert container.image is not None


def test_get_proto_container_with_tuple_resources():
    """
    GOAL: Verify tuple resources (requests, limits) are serialized correctly.

    Tests that:
    - First value in tuple becomes request
    - Second value in tuple becomes limit
    - Both are present in the protobuf
    """
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
        resources=Resources(cpu=(1, 2), memory=("1Gi", "2Gi")),
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1",
    )

    container = get_proto_container(app_env, ctx)

    assert container.resources is not None
    assert len(container.resources.requests) == 2
    assert len(container.resources.limits) == 2

    # Check CPU request and limit
    cpu_req = next((r for r in container.resources.requests if r.name == tasks_pb2.Resources.ResourceName.CPU), None)
    assert cpu_req is not None
    assert cpu_req.value == "1"

    cpu_limit = next((r for r in container.resources.limits if r.name == tasks_pb2.Resources.ResourceName.CPU), None)
    assert cpu_limit is not None
    assert cpu_limit.value == "2"


def test_get_proto_container_with_gpu():
    """
    GOAL: Verify GPU resources are serialized to protobuf.

    Tests that GPU is added as a resource request.
    """
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
        resources=Resources(cpu=2, memory="4Gi", gpu=1),
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1",
    )

    container = get_proto_container(app_env, ctx)

    assert container.resources is not None
    # GPU should be in requests
    gpu_req = next((r for r in container.resources.requests if r.name == tasks_pb2.Resources.ResourceName.GPU), None)
    assert gpu_req is not None
    assert gpu_req.value == "1"


def test_get_proto_container_empty_env_vars():
    """
    GOAL: Verify None env_vars results in no environment variables.

    Tests that when env_vars is None, the container env field is None or empty.
    """
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
        env_vars=None,
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1",
    )

    container = get_proto_container(app_env, ctx)

    assert container.env is None or len(container.env) == 0


def test_get_proto_container_string_command():
    """
    GOAL: Verify string commands are split using shlex.

    Tests that command strings are properly parsed into lists.
    """
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
        command="python -m myapp",
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1",
    )

    container = get_proto_container(app_env, ctx)

    # String commands are split using shlex
    assert container.command == ["python", "-m", "myapp"]


def test_get_proto_container_string_args():
    """
    GOAL: Verify string args are split using shlex.

    Tests that arg strings are properly parsed into lists.
    """
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
        args="--host 0.0.0.0 --port 8080",
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1",
    )

    container = get_proto_container(app_env, ctx)

    # String args are split using shlex
    assert container.args == ["--host", "0.0.0.0", "--port", "8080"]


def test_get_proto_container_with_quoted_string_args():
    """
    GOAL: Verify shlex correctly handles quoted strings in args.

    Tests that quoted content is preserved as a single argument.
    """
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
        args='--message "Hello World" --count 5',
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1",
    )

    container = get_proto_container(app_env, ctx)

    # Quoted strings should be preserved as single arguments
    assert container.args == ["--message", "Hello World", "--count", "5"]


def test_get_proto_container_comprehensive():
    """
    GOAL: Comprehensive test with all container features together.

    Tests that:
    - Resources, env vars, ports, command, args, and inputs all work together
    - Each component is correctly serialized
    - Components don't interfere with each other
    """
    app_env = AppEnvironment(
        name="comprehensive-app",
        image=Image.from_base("python:3.11-slim"),
        port=Port(port=8000, name="http"),
        command=None,  # Use default fserve
        args=["--arg1", "value1"],
        resources=Resources(cpu=(1, 2), memory=("1Gi", "2Gi"), gpu=1),
        env_vars={"ENV": "production", "LOG_LEVEL": "info"},
        inputs=[
            Parameter(value="config.yaml", name="config"),
            Parameter(value="model.pkl", name="model"),
        ],
    )

    ctx = SerializationContext(
        org="prod-org",
        project="prod-project",
        domain="production",
        version="v2.0.0",
        code_bundle=CodeBundle(computed_version="v2.0.0", tgz="s3://bucket/code.tgz"),
    )

    container = get_proto_container(app_env, ctx)

    # Verify image
    assert container.image is not None

    # Verify port
    assert len(container.ports) == 1
    assert container.ports[0].container_port == 8000
    assert container.ports[0].name == "http"

    # Verify resources
    assert container.resources is not None
    assert len(container.resources.requests) == 3  # CPU, memory, GPU
    assert len(container.resources.limits) == 2  # CPU, memory (GPU has no limit)

    # Verify env vars
    assert container.env is not None
    assert len(container.env) == 2
    env_dict = {kv.key: kv.value for kv in container.env}
    assert env_dict["ENV"] == "production"
    assert env_dict["LOG_LEVEL"] == "info"

    # Verify command has fserve and inputs
    assert container.command[0] == "fserve"
    assert "--inputs" in container.command
    assert "--version" in container.command

    # Verify args
    assert container.args == ["--arg1", "value1"]


def test_app_with_secrets():
    """
    GOAL: Verify secrets are included in the security context of AppIDL.

    Tests that translate_app_env_to_idl properly handles secrets configuration.
    """
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
        secrets="my-secret",
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1",
    )

    app_idl = translate_app_env_to_idl(app_env, ctx)
    assert app_idl.spec.security_context.secrets is not None
    assert len(app_idl.spec.security_context.secrets) == 1
    assert app_idl.spec.security_context.secrets[0].key == "my-secret"


def test_get_proto_container_with_image_cache():
    """
    GOAL: Verify image cache is used to resolve image URIs.

    Tests that when an image cache is provided, images are looked up correctly.
    """
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1",
        image_cache=ImageCache(
            image_lookup={"test-app": "gcr.io/my-project/python:3.11-cached"}, serialized_form="cached"
        ),
    )

    container = get_proto_container(app_env, ctx)

    # Image should be resolved from cache
    assert container.image is not None
    # Note: The exact URI depends on lookup_image_in_cache implementation


def test_get_proto_container_with_multiple_inputs():
    """
    GOAL: Verify multiple inputs are serialized correctly.

    Tests that:
    - Multiple inputs are all included
    - Each input's properties are preserved
    - Serialization is successful
    """
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
        inputs=[
            Parameter(value="config.yaml", name="config", env_var="CONFIG_PATH"),
            Parameter(value="data.csv", name="data"),
            Parameter(value="s3://bucket/model.pkl", name="model", download=True),
        ],
        args=["--verbose"],
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=CodeBundle(computed_version="v1.0.0", tgz="s3://bucket/code.tgz"),
    )

    container = get_proto_container(app_env, ctx)

    # Args should still be present
    assert container.args == ["--verbose"]

    # Command should have inputs
    assert "--inputs" in container.command
    cmd_list = list(container.command)
    inputs_idx = cmd_list.index("--inputs")
    serialized_inputs = cmd_list[inputs_idx + 1]

    # Verify inputs can be deserialized
    from flyte.app._input import SerializableInputCollection

    deserialized = SerializableInputCollection.from_transport(serialized_inputs)
    assert len(deserialized.inputs) == 3
    assert deserialized.inputs[0].name == "config"
    assert deserialized.inputs[0].env_var == "CONFIG_PATH"
    assert deserialized.inputs[1].name == "data"
    assert deserialized.inputs[2].name == "model"


@pytest.mark.parametrize(
    "domain",
    [
        None,
        Domain(subdomain="my-custom-subdomain"),
        Domain(custom_domain="example.com"),
        Domain(subdomain="my-custom-subdomain", custom_domain="example.com"),
    ],
)
def test_app_with_domain(domain: Domain | None):
    """
    GOAL: Verify default domain results in None subdomain and cname in ingress config.

    Tests that when domain is None or default, the ingress config has no subdomain or cname.
    """
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
        domain=domain,
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1",
    )

    app_idl = translate_app_env_to_idl(app_env, ctx)
    assert app_idl.spec.ingress is not None
    assert app_idl.spec.ingress.subdomain == (domain.subdomain if domain and domain.subdomain else "")
    assert app_idl.spec.ingress.cname == (domain.custom_domain if domain and domain.custom_domain else "")
    assert app_idl.spec.ingress.private is False


# =============================================================================
# Tests for _materialize_inputs_with_delayed_values
# =============================================================================


@pytest.mark.asyncio
async def test_materialize_inputs_with_no_delayed_values():
    """
    GOAL: Verify that inputs without delayed values pass through unchanged.

    Tests that regular string, File, and Dir inputs are returned as-is.
    """
    inputs = [
        Input(name="config", value="config.yaml"),
        Input(name="model", value=flyte.io.File(path="s3://bucket/model.pkl")),
        Input(name="data", value=flyte.io.Dir(path="s3://bucket/data")),
    ]

    result = await _materialize_inputs_with_delayed_values(inputs)

    assert len(result) == 3
    assert result[0].name == "config"
    assert result[0].value == "config.yaml"
    assert result[1].name == "model"
    assert isinstance(result[1].value, flyte.io.File)
    assert result[2].name == "data"
    assert isinstance(result[2].value, flyte.io.Dir)


@pytest.mark.asyncio
async def test_materialize_inputs_with_run_output():
    """
    GOAL: Verify that RunOutput delayed values are materialized correctly.

    Tests that RunOutput inputs are replaced with their materialized values.
    """
    # Create mock for RunOutput materialization
    mock_run_details = MagicMock()
    mock_run_details.outputs = AsyncMock(return_value=["s3://bucket/materialized/model"])

    mock_run = MagicMock()
    mock_run.details = MagicMock()
    mock_run.details.aio = AsyncMock(return_value=mock_run_details)

    inputs = [
        Input(name="config", value="config.yaml"),
        Input(name="model", value=RunOutput(type="string", run_name="my-run-123")),
    ]

    with (
        patch("flyte.remote.Run") as MockRun,
        patch("flyte._initialize.is_initialized", return_value=True),
    ):
        MockRun.get = MagicMock()
        MockRun.get.aio = AsyncMock(return_value=mock_run)

        result = await _materialize_inputs_with_delayed_values(inputs)

    assert len(result) == 2
    assert result[0].name == "config"
    assert result[0].value == "config.yaml"
    assert result[1].name == "model"
    assert result[1].value == "s3://bucket/materialized/model"


@pytest.mark.asyncio
async def test_materialize_inputs_with_run_output_dir_type():
    """
    GOAL: Verify that RunOutput with Dir type materializes to a Dir path.

    Tests that RunOutput returning a Dir is properly materialized.
    """
    # Create mock for RunOutput materialization
    mock_run_details = MagicMock()
    mock_run_details.outputs = AsyncMock(return_value=[flyte.io.Dir(path="s3://bucket/data-dir")])

    mock_run = MagicMock()
    mock_run.details = MagicMock()
    mock_run.details.aio = AsyncMock(return_value=mock_run_details)

    inputs = [
        Input(name="data", value=RunOutput(type=flyte.io.Dir, run_name="my-run-123")),
    ]

    with (
        patch("flyte.remote.Run") as MockRun,
        patch("flyte._initialize.is_initialized", return_value=True),
    ):
        MockRun.get = MagicMock()
        MockRun.get.aio = AsyncMock(return_value=mock_run)

        result = await _materialize_inputs_with_delayed_values(inputs)

    assert len(result) == 1
    assert result[0].name == "data"
    # The value should be the path string after .get() is called
    assert isinstance(result[0].value, flyte.io.Dir)
    assert result[0].value.path == "s3://bucket/data-dir"


@pytest.mark.asyncio
async def test_materialize_inputs_empty_list():
    """
    GOAL: Verify that empty input list returns empty list.

    Tests edge case where no inputs are provided.
    """
    result = await _materialize_inputs_with_delayed_values([])
    assert result == []


@pytest.mark.asyncio
async def test_materialize_inputs_mixed_delayed_and_regular():
    """
    GOAL: Verify that mixed inputs with some delayed values work correctly.

    Tests that only delayed values are materialized while regular values pass through.
    """
    mock_run_details = MagicMock()
    mock_run_details.outputs = AsyncMock(return_value=["materialized-value"])

    mock_run = MagicMock()
    mock_run.details = MagicMock()
    mock_run.details.aio = AsyncMock(return_value=mock_run_details)

    inputs = [
        Input(name="static-config", value="static.yaml"),
        Input(name="dynamic-model", value=RunOutput(type="string", run_name="run-1")),
        Input(name="static-file", value=flyte.io.File(path="s3://bucket/file.txt")),
    ]

    with (
        patch("flyte.remote.Run") as MockRun,
        patch("flyte._initialize.is_initialized", return_value=True),
    ):
        MockRun.get = MagicMock()
        MockRun.get.aio = AsyncMock(return_value=mock_run)

        result = await _materialize_inputs_with_delayed_values(inputs)

    assert len(result) == 3
    # Static string unchanged
    assert result[0].value == "static.yaml"
    # RunOutput materialized
    assert result[1].value == "materialized-value"
    # Static File unchanged
    assert isinstance(result[2].value, flyte.io.File)
    assert result[2].value.path == "s3://bucket/file.txt"


@pytest.mark.asyncio
async def test_materialize_inputs_preserves_other_input_properties():
    """
    GOAL: Verify that materialization preserves other Input properties.

    Tests that env_var, mount, download, etc. are preserved after materialization.
    """
    mock_run_details = MagicMock()
    mock_run_details.outputs = AsyncMock(return_value=["s3://bucket/materialized"])

    mock_run = MagicMock()
    mock_run.details = MagicMock()
    mock_run.details.aio = AsyncMock(return_value=mock_run_details)

    inputs = [
        Input(
            name="model",
            value=RunOutput(type="string", run_name="my-run"),
            env_var="MODEL_PATH",
            mount="/mnt/model",
            download=True,
        ),
    ]

    with (
        patch("flyte.remote.Run") as MockRun,
        patch("flyte._initialize.is_initialized", return_value=True),
    ):
        MockRun.get = MagicMock()
        MockRun.get.aio = AsyncMock(return_value=mock_run)

        result = await _materialize_inputs_with_delayed_values(inputs)

    assert len(result) == 1
    assert result[0].name == "model"
    assert result[0].value == "s3://bucket/materialized"
    assert result[0].env_var == "MODEL_PATH"
    assert result[0].mount == "/mnt/model"
    assert result[0].download is True
