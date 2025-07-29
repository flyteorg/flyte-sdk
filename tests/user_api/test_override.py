import pytest

import flyte

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
