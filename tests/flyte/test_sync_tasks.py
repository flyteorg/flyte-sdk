from typing import List

import flyte

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
    assert result.outputs() == ["Hello, world 0!", "Hello, world 1!", "Hello, world 2!"]
