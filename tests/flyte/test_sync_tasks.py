from typing import List

import pytest

import flyte
from flyte.errors import RuntimeUserError, SyncTaskInAsyncContextError

env = flyte.TaskEnvironment(name="test")


@env.task
def sync_task1(v: str) -> str:
    return f"Hello, world {v}!"


@env.task
def sync_parent_task(i: int) -> List[str]:
    vals = []
    for i in range(i):
        vals.append(sync_task1(str(i)))
    return vals


def test_parent_action_raw():
    result = sync_parent_task(3)
    assert result == ["Hello, world 0!", "Hello, world 1!", "Hello, world 2!"]


def test_typing():
    assert sync_parent_task._call_as_synchronous is True


def test_parent_action_local():
    flyte.init()
    result = flyte.run(sync_parent_task, 3)
    assert result.outputs()[0] == ["Hello, world 0!", "Hello, world 1!", "Hello, world 2!"]


@env.task
async def async_parent_calling_sync_child(i: int) -> str:
    return sync_task1(str(i))


@env.task
async def async_parent_awaiting_sync_child(i: int) -> str:
    return await sync_task1.aio(str(i))


def test_sync_task_in_async_context_error_is_user_error():
    assert issubclass(SyncTaskInAsyncContextError, RuntimeUserError)


def test_sync_child_from_async_parent_raises():
    flyte.init()
    with pytest.raises(RuntimeUserError, match="aio"):
        flyte.run(async_parent_calling_sync_child, 1).outputs()


def test_sync_child_from_async_parent_with_aio_succeeds():
    flyte.init()
    result = flyte.run(async_parent_awaiting_sync_child, 1)
    assert result.outputs()[0] == "Hello, world 1!"
