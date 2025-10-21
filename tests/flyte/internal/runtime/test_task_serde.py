import pathlib

import pytest
from flyteidl2.core import identifier_pb2, interface_pb2, literals_pb2, tasks_pb2, types_pb2
from flyteidl2.core.security_pb2 import Secret as ProtoSecret
from flyteidl2.core.security_pb2 import SecurityContext
from flyteidl2.task import common_pb2, environment_pb2
from kubernetes.client import (
    V1Container,
    V1EnvVar,
    V1LocalObjectReference,
    V1PodSpec,
)

import flyte
from flyte import PodTemplate
from flyte._internal.runtime.task_serde import (
    _get_k8s_pod,
    _get_urun_container,
    get_proto_task,
    get_security_context,
    translate_task_to_wire,
)
from flyte._secret import Secret
from flyte.models import SerializationContext


def test_get_security_context():
    # Case 1: No secrets provided
    assert get_security_context(None) is None

    # Case 2: Single secret with environment variable
    secrets = Secret(group="group1", key="key1", as_env_var="ENV_VAR1")
    security_context = get_security_context(secrets)
    assert isinstance(security_context, SecurityContext)
    assert len(security_context.secrets) == 1
    assert security_context.secrets[0].group == "group1"
    assert security_context.secrets[0].key == "key1"
    assert security_context.secrets[0].mount_requirement == ProtoSecret.MountType.ENV_VAR
    assert security_context.secrets[0].env_var == "ENV_VAR1"

    # Case 3: Single secret with file mount
    secrets = Secret(group="group2", key="key2", as_env_var=None)
    security_context = get_security_context(secrets)
    assert isinstance(security_context, SecurityContext)
    assert len(security_context.secrets) == 1
    assert security_context.secrets[0].group == "group2"
    assert security_context.secrets[0].key == "key2"
    assert security_context.secrets[0].mount_requirement == ProtoSecret.MountType.ENV_VAR
    assert security_context.secrets[0].env_var == "GROUP2_KEY2"

    # Case 4: Multiple secrets
    secrets = [
        Secret(group="group1", key="key1", as_env_var="ENV_VAR1"),
        Secret(group="group2", key="key2", as_env_var=None),
    ]
    security_context = get_security_context(secrets)
    assert isinstance(security_context, SecurityContext)
    assert len(security_context.secrets) == 2
    assert security_context.secrets[0].group == "group1"
    assert security_context.secrets[0].key == "key1"
    assert security_context.secrets[0].mount_requirement == ProtoSecret.MountType.ENV_VAR
    assert security_context.secrets[0].env_var == "ENV_VAR1"
    assert security_context.secrets[1].group == "group2"
    assert security_context.secrets[1].key == "key2"
    assert security_context.secrets[1].mount_requirement == ProtoSecret.MountType.ENV_VAR
    assert security_context.secrets[1].env_var == "GROUP2_KEY2"

    # Case 5: Invalid secret input (not a Secret or list of Secrets)
    with pytest.raises(AttributeError):
        get_security_context(["invalid_secret", 1])


def test_get_proto_container_task():
    # Create a real task environment
    env = flyte.TaskEnvironment(
        name="test_env",
        image="python:3.10",
        resources=flyte.Resources(cpu="1", memory="2Gi"),
        env_vars={"ENV1": "val1", "ENV2": "val2"},
    )

    # Create a task using the environment
    @env.task(
        short_name="real_test_task",
        cache=flyte.Cache(behavior="auto"),
        retries=3,
        timeout=60,
    )
    async def t1(a: int, b: str) -> str:
        """Test function docstring"""
        return f"{a} {b}"

    # Get the task template from the decorated function
    task_template = t1

    # Create serialization context
    context = SerializationContext(
        project="test-project",
        domain="test-domain",
        version="test-version",
        org="test-org",
        input_path="/tmp/inputs",
        output_path="/tmp/outputs",
        image_cache=None,
        code_bundle=None,
        root_dir=pathlib.Path.cwd(),
    )

    # Generate proto task
    proto_task = get_proto_task(task_template, context)

    # Verify the proto task properties
    assert isinstance(proto_task, tasks_pb2.TaskTemplate)

    # Check identifier
    assert proto_task.id.resource_type == identifier_pb2.ResourceType.TASK
    assert proto_task.id.project == "test-project"
    assert proto_task.id.domain == "test-domain"
    assert proto_task.id.name == "test_env.t1"
    assert proto_task.id.version == "test-version"
    assert proto_task.id.org == "test-org"

    # Check basic properties
    assert proto_task.type == "python"
    assert proto_task.task_type_version == 0

    # Check metadata
    assert proto_task.metadata.discoverable is True  # Cache is enabled
    assert proto_task.metadata.retries.retries == 3
    assert proto_task.metadata.timeout.seconds == 60

    # Check interface
    assert proto_task.interface.inputs.variables["a"].type.simple == types_pb2.INTEGER  # Integer
    assert proto_task.interface.inputs.variables["b"].type.simple == types_pb2.STRING  # String
    assert proto_task.interface.outputs.variables["o0"].type.simple == types_pb2.STRING  # String

    # Check container
    assert proto_task.container.image.startswith("python:3.10")
    assert "a0" in " ".join(proto_task.container.args)
    assert len(proto_task.container.env) >= 2

    # Environment variables should be present
    env_vars = {env_var.key: env_var.value for env_var in proto_task.container.env}
    assert "ENV1" in env_vars
    assert env_vars["ENV1"] == "val1"
    assert "ENV2" in env_vars
    assert env_vars["ENV2"] == "val2"


def test_get_proto_task_ignored_cache_inputs():
    # Create a real task environment
    env = flyte.TaskEnvironment(
        name="test_env_cache",
        image="python:3.10",
        resources=flyte.Resources(cpu="1", memory="2Gi"),
        env_vars={"ENV1": "val1", "ENV2": "val2"},
    )

    # Create a task using the environment
    @env.task(
        short_name="real_test_task",
        cache=flyte.Cache(behavior="auto", ignored_inputs="my_ignored_input"),
        retries=3,
        timeout=60,
    )
    async def test_function(a: int, my_ignored_input: str) -> str:
        """Test function docstring"""
        return f"{a} {my_ignored_input}"

    # Get the task template from the decorated function
    task_template = test_function

    # Create serialization context
    context = SerializationContext(
        project="test-project",
        domain="test-domain",
        version="test-version",
        org="test-org",
        input_path="/tmp/inputs",
        output_path="/tmp/outputs",
        image_cache=None,
        code_bundle=None,
        root_dir=pathlib.Path.cwd(),
    )

    # Generate proto task
    proto_task = get_proto_task(task_template, context)

    # Verify the proto task properties
    assert isinstance(proto_task, tasks_pb2.TaskTemplate)
    assert proto_task.metadata.cache_ignore_input_vars == ["my_ignored_input"]


def test_get_proto_k8s_pod_task():
    pod_template1 = PodTemplate(
        pod_spec=V1PodSpec(
            containers=[V1Container(name="primary", env=[V1EnvVar(name="hello", value="world")])],
            image_pull_secrets=[V1LocalObjectReference(name="regcred-test")],
        ),
        labels={"foo": "bar"},
        annotations={"baz": "qux"},
    )

    env = flyte.TaskEnvironment(
        name="test_env",
        image="python:3.10",
        resources=flyte.Resources(cpu="1", memory="2Gi"),
        env_vars={"ENV1": "val1", "ENV2": "val2"},
        pod_template=pod_template1,
    )

    @env.task(
        short_name="real_test_task",
    )
    async def t1(a: int, b: str) -> str:
        """Test function docstring"""
        return f"{a} {b}"

    # Get the task template from the decorated function
    task_template = t1

    # Create serialization context
    context = SerializationContext(
        project="test-project",
        domain="test-domain",
        version="test-version",
        org="test-org",
        input_path="/tmp/inputs",
        output_path="/tmp/outputs",
        image_cache=None,
        code_bundle=None,
        root_dir=pathlib.Path.cwd(),
    )

    # Generate proto task
    proto_task = get_proto_task(task_template, context)

    # Verify the proto task properties
    assert isinstance(proto_task, tasks_pb2.TaskTemplate)

    # Check k8s_pod
    k8s_pod = _get_k8s_pod(_get_urun_container(context, t1), pod_template1)
    assert proto_task.k8s_pod == k8s_pod
    assert proto_task.k8s_pod.metadata.labels == {"foo": "bar"}
    assert proto_task.k8s_pod.metadata.annotations == {"baz": "qux"}

    pod_template2 = PodTemplate(
        pod_spec=V1PodSpec(
            containers=[V1Container(name="foo", env=[V1EnvVar(name="hello", value="world")])],
            image_pull_secrets=[V1LocalObjectReference(name="regcred-test")],
        )
    )

    env = flyte.TaskEnvironment(
        name="test_env",
        pod_template=pod_template2,
    )

    @env.task(
        short_name="real_test_task",
    )
    async def t2(a: int, b: str) -> str:
        """Test function docstring"""
        return f"{a} {b}"

    with pytest.raises(ValueError):
        get_proto_task(t2, context)


@pytest.fixture(scope="module")
def env_task_ctx():
    # Create a real task environment
    env = flyte.TaskEnvironment(
        name="test_env",
        image="python:3.10",
        resources=flyte.Resources(cpu="1", memory="2Gi"),
        env_vars={"ENV1": "val1", "ENV2": "val2"},
    )

    # Create a task using the environment
    @env.task(
        short_name="real_test_task",
        cache=flyte.Cache(behavior="auto"),
        retries=3,
        timeout=60,
    )
    async def t1(a: int, b: str) -> str:
        """Test function docstring"""
        return f"{a} {b}"

    # Get the task template from the decorated function
    task_template = t1

    # Create serialization context
    context = SerializationContext(
        project="test-project",
        domain="test-domain",
        version="test-version",
        org="test-org",
        input_path="/tmp/inputs",
        output_path="/tmp/outputs",
        image_cache=None,
        code_bundle=None,
        root_dir=pathlib.Path.cwd(),
    )
    return env, task_template, context


def test_translate_task_to_wire(env_task_ctx):
    _env, task_template, context = env_task_ctx
    # Generate proto task
    proto_task = translate_task_to_wire(task_template, context)

    assert proto_task.task_template.id.project == "test-project"
    assert proto_task.task_template.id.domain == "test-domain"
    assert proto_task.task_template.id.name == "test_env.t1"
    assert proto_task.task_template.id.version == "test-version"
    assert proto_task.short_name == "real_test_task"
    assert proto_task.environment == environment_pb2.Environment(name="test_env")
    assert proto_task.default_inputs == []


def test_translate_task_to_wire_with_default_inputs(env_task_ctx):
    _env, task_template, context = env_task_ctx

    default_inputs = [
        common_pb2.NamedParameter(
            name="a",
            parameter=interface_pb2.Parameter(
                var=interface_pb2.Variable(
                    type=types_pb2.LiteralType(simple=types_pb2.INTEGER),
                ),
                default=literals_pb2.Literal(
                    scalar=literals_pb2.Scalar(
                        primitive=literals_pb2.Primitive(integer=10),
                    ),
                ),
            ),
        ),
        common_pb2.NamedParameter(
            name="b",
            parameter=interface_pb2.Parameter(
                var=interface_pb2.Variable(
                    type=types_pb2.LiteralType(simple=types_pb2.STRING),
                ),
                default=literals_pb2.Literal(
                    scalar=literals_pb2.Scalar(
                        primitive=literals_pb2.Primitive(string_value="default"),
                    ),
                ),
            ),
        ),
    ]
    # Generate proto task
    proto_task = translate_task_to_wire(task_template, context, default_inputs=default_inputs)

    assert proto_task.default_inputs == default_inputs
