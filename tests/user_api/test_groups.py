import pytest

import flyte
import flyte.errors

env = flyte.TaskEnvironment("test")


@env.task
async def task1():
    """
    A test task that does nothing.
    """
    with flyte.group("test_group"):
        assert flyte.ctx().group_data
        assert flyte.ctx().group_data.name == "test_group"


@pytest.mark.asyncio
async def test_group():
    """
    Test the group context manager.
    """
    with pytest.raises(Exception):
        await task1()


@pytest.mark.asyncio
async def test_group_with_run():
    """
    Test the group context manager with runcontext.
    """
    await flyte.init.aio(api_key="")
    await flyte.run.aio(task1)
    assert flyte.ctx() is None
