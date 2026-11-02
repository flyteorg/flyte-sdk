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
    assert not flyte.ctx()


def test_group_name_too_long():
    """Group names over the control plane's 255-character limit are rejected before anything runs."""
    from flyte.models import MAX_GROUP_NAME_LENGTH

    with flyte.group("g" * MAX_GROUP_NAME_LENGTH):
        pass
    with pytest.raises(ValueError, match="maximum is 255"):
        with flyte.group("g" * (MAX_GROUP_NAME_LENGTH + 1)):
            pass
