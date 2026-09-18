"""copy.copy vs pickle on a TaskTemplate: a same-process copy keeps parent_env, a pickle drops it."""

import copy

import cloudpickle

import flyte


def _make_task():
    env = flyte.TaskEnvironment(name="copy_env", image="python:3.11")

    @env.task
    async def t(x: int) -> int:
        return x

    return env, t


def test_copy_preserves_parent_env():
    env, t = _make_task()
    c = copy.copy(t)
    assert c is not t
    assert c.parent_env is not None
    assert c.parent_env() is env
    assert c.parent_env_name == "copy_env"
    # Still a shallow copy: the deploy sys-path copy mutates env_vars on the copy only.
    c.env_vars = {"A": "1"}
    assert t.env_vars is None


def test_pickle_drops_parent_env():
    _, t = _make_task()
    restored = cloudpickle.loads(cloudpickle.dumps(t))
    assert restored.parent_env is None
    assert restored.parent_env_name == "copy_env"
