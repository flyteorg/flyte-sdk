import pathlib

import pytest

import flyte
from flyte._internal.runtime.task_serde import translate_task_to_wire
from flyte._namespace import TASK_NAMESPACE_CONFIG_KEY
from flyte.models import SerializationContext


def test_task_namespace_configuration_and_serialization():
    env = flyte.TaskEnvironment(name="namespace_env", image="python:3.12", namespace="env-workers")

    @env.task
    async def inherited() -> None:
        pass

    @env.task(namespace="gpu-workers")
    async def explicit() -> None:
        pass

    assert inherited.namespace == "env-workers"
    assert explicit.namespace == "gpu-workers"
    assert explicit.override(namespace="other-workers").namespace == "other-workers"
    assert explicit.namespace == "gpu-workers"
    assert explicit.override(retries=1).namespace == "gpu-workers"
    assert env.clone_with("cloned", namespace="clone-workers").namespace == "clone-workers"
    assert env.clone_with("inherited_clone").namespace == "env-workers"

    context = SerializationContext(
        project="project", domain="development", version="version", root_dir=pathlib.Path.cwd()
    )
    for task, namespace in [
        (inherited, "env-workers"),
        (explicit, "gpu-workers"),
        (explicit.override(namespace="other-workers"), "other-workers"),
    ]:
        wire = translate_task_to_wire(task, context)
        assert wire.task_template.config[TASK_NAMESPACE_CONFIG_KEY] == namespace


def test_task_namespace_unset_keeps_existing_wire_format():
    env = flyte.TaskEnvironment(name="default_namespace_env", image="python:3.12")

    @env.task
    async def task() -> None:
        pass

    context = SerializationContext(
        project="project", domain="development", version="version", root_dir=pathlib.Path.cwd()
    )
    assert TASK_NAMESPACE_CONFIG_KEY not in translate_task_to_wire(task, context).task_template.config


@pytest.mark.parametrize("namespace", ["", "Uppercase", "with_underscore", "a.b", "-prefix", "suffix-", "a" * 64])
def test_invalid_task_namespace_rejected(namespace):
    with pytest.raises(ValueError, match="Kubernetes namespace"):
        flyte.TaskEnvironment(name="invalid_namespace", namespace=namespace)

    env = flyte.TaskEnvironment(name="invalid_task_namespace")
    with pytest.raises(ValueError, match="Kubernetes namespace"):

        @env.task(namespace=namespace)
        async def task() -> None:
            pass


@pytest.mark.parametrize("namespace", [42, [], {}])
def test_namespace_must_be_string(namespace):
    with pytest.raises(TypeError, match="namespace"):
        flyte.TaskEnvironment(name="invalid_namespace_type", namespace=namespace)


def test_reusable_namespace_override_requires_disabling_reuse():
    env = flyte.TaskEnvironment(name="reusable_namespace", namespace="gpu-workers", reusable=flyte.ReusePolicy())

    @env.task
    async def task() -> None:
        pass

    with pytest.raises(ValueError, match="Cannot override namespace"):
        task.override(namespace="other-workers")
    assert task.override(namespace="other-workers", reusable="off").namespace == "other-workers"


def test_container_task_namespace_serializes_and_survives_override():
    from flyte.extras import ContainerTask

    task = ContainerTask(
        name="run_training_container",
        image="training-image:latest",
        command=["python", "train.py"],
        namespace="gpu-workers",
    )
    flyte.TaskEnvironment.from_task("container_namespace_env", task)
    context = SerializationContext(
        project="project", domain="development", version="version", root_dir=pathlib.Path.cwd()
    )
    assert translate_task_to_wire(task, context).task_template.config[TASK_NAMESPACE_CONFIG_KEY] == "gpu-workers"
    assert task.override(namespace="other-workers").namespace == "other-workers"
    assert task.namespace == "gpu-workers"
