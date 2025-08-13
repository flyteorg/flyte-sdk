import pathlib

import pytest

import flyte
from flyte import RetryStrategy
from flyte._internal.runtime.task_serde import get_proto_task, get_security_context
from flyte._protos.workflow import task_definition_pb2
from flyte.models import SerializationContext
from flyte.remote._task import TaskDetails

env = flyte.TaskEnvironment(name="hello_world", resources=flyte.Resources(cpu=1, memory="250Mi"))


@env.task
async def oomer(x: int) -> int:
    pass


env_with_reuse = flyte.TaskEnvironment(
    name="oomer_with_reuse",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
    reusable=flyte.ReusePolicy(replicas=2, idle_ttl=60),
)


@env_with_reuse.task
async def oomer_with_reuse(x: int) -> int:
    pass


def test_oomer_override():
    """
    Test the override functionality of the oomer task.
    """
    # Create a new task with overridden resources
    new_task = oomer.override(resources=flyte.Resources(cpu=2, memory="500Mi"))

    # Check if the new task has the correct resources
    assert new_task.resources.cpu == 2
    assert new_task.resources.memory == "500Mi"
    assert isinstance(new_task.cache, flyte.Cache)

    # Check if the new task is not the same as the original task
    assert new_task != oomer


def test_oomer_override_with_reuse_incorrect():
    """
    Test the override functionality of the oomer task with reuse.
    """
    # Create a new task with overridden resources and reuse policy
    with pytest.raises(ValueError):
        oomer.override(
            resources=flyte.Resources(cpu=2, memory="500Mi"),
            reusable=flyte.ReusePolicy(replicas=2, idle_ttl=60),
        )

    with pytest.raises(ValueError):
        oomer_with_reuse.override(
            resources=flyte.Resources(cpu=2, memory="500Mi"),
        )

    with pytest.raises(ValueError):
        oomer_with_reuse.override(
            env={},
        )

    with pytest.raises(ValueError):
        oomer_with_reuse.override(
            secrets="my_secret",
        )


def test_override_with_reuse():
    """
    Test the override functionality of the oomer task with reuse.
    """
    # Create a new task with overridden resources and reuse policy
    new_task = oomer_with_reuse.override(
        cache=flyte.Cache("auto"),
    )

    # Check if the new task has the correct resources
    assert new_task.resources.cpu == 1
    assert new_task.resources.memory == "250Mi"
    assert isinstance(new_task.cache, flyte.Cache)

    # Check if the new task is not the same as the original task
    assert new_task != oomer_with_reuse


def test_override_turn_reuse_off():
    """
    Test the override functionality of the oomer task with reuse turned off.
    """
    # Create a new task with reuse turned off
    new_task = oomer_with_reuse.override(reusable="off", resources=flyte.Resources(cpu=2, memory="500Mi"))

    # Check if the new task has the correct resources
    assert new_task.resources.cpu == 2
    assert new_task.resources.memory == "500Mi"
    assert new_task.reusable is None

    # Check if the new task is not the same as the original task
    assert new_task != oomer_with_reuse


def test_override_ref_task():
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
    new_task = oomer_with_reuse.override(reusable="off", resources=flyte.Resources(cpu=2, memory="500Mi"))
    task_template = get_proto_task(new_task, context)

    task_details_pb2 = task_definition_pb2.TaskDetails(spec=task_definition_pb2.TaskSpec(task_template=task_template))
    td = TaskDetails(pb2=task_details_pb2)

    secrets = [flyte.Secret(key="openai", as_env_var="OPENAI_API_KEY")]
    td.override(
        resources=flyte.Resources(cpu=3, memory="100Mi"),
        retries=RetryStrategy(5),
        timeout=100,
        env={"FOO": "BAR"},
        secrets=secrets,
    )
    assert td.resources[0][0].value == "3"
    assert td.resources[0][1].value == "100Mi"
    assert td.pb2.spec.task_template.metadata.retries.retries == 5
    assert td.pb2.spec.task_template.metadata.timeout.seconds == 100
    assert td.pb2.spec.task_template.security_context == get_security_context(secrets)
