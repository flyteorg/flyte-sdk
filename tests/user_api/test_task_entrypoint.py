from __future__ import annotations

import flyte
from flyte.extras._container import ContainerTask


def test_task_template_entrypoint_default_false():
    """Tasks should have entrypoint=False by default."""
    env = flyte.TaskEnvironment(name="test_default", image="python:3.10")

    @env.task()
    async def my_task(x: int) -> int:
        return x

    assert my_task.entrypoint is False


def test_task_template_entrypoint_true():
    """Tasks created with entrypoint=True should have it set."""
    env = flyte.TaskEnvironment(name="test_true", image="python:3.10")

    @env.task(entrypoint=True)
    async def my_task(x: int) -> int:
        return x

    assert my_task.entrypoint is True


def test_task_override_entrypoint():
    """Override should propagate the entrypoint flag."""
    env = flyte.TaskEnvironment(name="test_override", image="python:3.10")

    @env.task()
    async def my_task(x: int) -> int:
        return x

    assert my_task.entrypoint is False

    overridden = my_task.override(entrypoint=True)
    assert overridden.entrypoint is True

    # Original should remain unchanged
    assert my_task.entrypoint is False


def test_task_environment_passes_entrypoint():
    """@env.task(entrypoint=True) should set the flag on the resulting template."""
    env = flyte.TaskEnvironment(name="test_env_pass", image="python:3.10")

    @env.task(entrypoint=True)
    async def entry_task(x: int) -> int:
        return x

    @env.task()
    async def normal_task(x: int) -> int:
        return x

    assert entry_task.entrypoint is True
    assert normal_task.entrypoint is False


def test_container_task_entrypoint():
    """ContainerTask should accept entrypoint=True through **kwargs."""
    ct = ContainerTask(
        name="test_container",
        image="python:3.10",
        command=["echo", "hello"],
        entrypoint=True,
    )
    assert ct.entrypoint is True


def test_container_task_entrypoint_default():
    """ContainerTask should default to entrypoint=False."""
    ct = ContainerTask(
        name="test_container_default",
        image="python:3.10",
        command=["echo", "hello"],
    )
    assert ct.entrypoint is False
