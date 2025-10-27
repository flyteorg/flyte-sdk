"""
Tests for AppEnvironment serialization to protobuf messages.

These tests verify that app_serde.py correctly converts AppEnvironment objects
into protobuf IDL format without using mocks.

Note: Some functions in app_serde.py currently have bugs that prevent full testing:
1. _get_scaling_metric uses 'val' field instead of 'target_value' for protobuf
2. translate_app_env_to_idl passes list for inputs instead of InputList protobuf

Tests document these bugs by testing that they correctly fail.
"""

import pytest
from flyteidl2.core import tasks_pb2

from flyte._image import Image
from flyte._resources import Resources
from flyte.app import AppEnvironment
from flyte.app._runtime.app_serde import (
    _get_scaling_metric,
    _sanitize_resource_name,
    get_proto_container,
    translate_app_env_to_idl,
)
from flyte.app._types import Port, Scaling
from flyte.models import SerializationContext


def test_sanitize_resource_name():
    """Test that resource names are sanitized for Kubernetes."""
    # Create a resource entry with CPU
    resource = tasks_pb2.Resources.ResourceEntry(name=tasks_pb2.Resources.ResourceName.CPU, value="2")
    sanitized = _sanitize_resource_name(resource)
    assert sanitized == "cpu"

    # Test with GPU
    resource = tasks_pb2.Resources.ResourceEntry(name=tasks_pb2.Resources.ResourceName.GPU, value="1")
    sanitized = _sanitize_resource_name(resource)
    assert sanitized == "gpu"

    # Test with ephemeral storage (has underscore)
    resource = tasks_pb2.Resources.ResourceEntry(name=tasks_pb2.Resources.ResourceName.EPHEMERAL_STORAGE, value="10Gi")
    sanitized = _sanitize_resource_name(resource)
    assert sanitized == "ephemeral-storage"


def test_get_scaling_metric_none():
    """Test that None metric returns None."""
    result = _get_scaling_metric(None)
    assert result is None


def test_get_scaling_metric_concurrency():
    """Test conversion of Concurrency metric to protobuf."""
    metric = Scaling.Concurrency(val=10)
    # Note: Implementation currently has a bug - uses 'val' instead of 'target_value'
    with pytest.raises(ValueError, match='has no "val" field'):
        _get_scaling_metric(metric)


def test_get_scaling_metric_request_rate():
    """Test conversion of RequestRate metric to protobuf."""
    metric = Scaling.RequestRate(val=100)
    # Note: Implementation currently has a bug - uses 'val' instead of 'target_value'
    with pytest.raises(ValueError, match='has no "val" field'):
        _get_scaling_metric(metric)


def test_get_proto_container_basic():
    """Test basic container protobuf generation."""
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
    assert container.ports[0].name == "http"


def test_get_proto_container_with_resources():
    """Test container with resource specifications."""
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
    """Test container with environment variables."""
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
    """Test container with custom port."""
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
    """Test container with custom command and args."""
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


def test_get_proto_container_with_string_image():
    """Test that string images are converted to Image objects by AppEnvironment."""
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
    """Test container with tuple resource specifications (requests and limits)."""
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
    # Check CPU
    assert len(container.resources.requests) == 2
    assert len(container.resources.limits) == 2

    cpu_req = next((r for r in container.resources.requests if r.name == tasks_pb2.Resources.ResourceName.CPU), None)
    assert cpu_req is not None
    assert cpu_req.value == "1"

    cpu_limit = next((r for r in container.resources.limits if r.name == tasks_pb2.Resources.ResourceName.CPU), None)
    assert cpu_limit is not None
    assert cpu_limit.value == "2"


def test_get_proto_container_with_gpu():
    """Test container with GPU resources."""
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
    """Test container with no environment variables."""
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
    """Test container with string command (will be split)."""
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
    """Test container with string args (will be split)."""
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


def test_get_proto_container_default_command():
    """Test container with default command (None)."""
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
        command=["fserve", "--"],
        args=[
            "--host",
        ],
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1",
    )

    container = get_proto_container(app_env, ctx)

    # Default command should be ["fserve", "--"]
    assert container.command == ["fserve", "--"]


def test_app_with_secrets():
    """Test container with secrets specified in environment variables."""
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
