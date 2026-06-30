"""Tests for :class:`flyte._internal.resolvers.default.DefaultTaskResolver`.

The default resolver is intentionally minimal: it imports the module and returns
the module-level attribute as-is.
"""

from __future__ import annotations

from flyte import TaskEnvironment
from flyte._internal.resolvers.default import DefaultTaskResolver
from flyte._task import TaskTemplate

_env = TaskEnvironment(name="default_resolver_env", image="auto")


@_env.task
async def _real_task(x: int) -> int:
    """A plain task used to exercise resolution."""
    return x


def test_load_task_returns_plain_task_unchanged():
    resolver = DefaultTaskResolver()
    loaded = resolver.load_task(["mod", __name__, "instance", "_real_task"])
    assert isinstance(loaded, TaskTemplate)
    assert loaded is _real_task
