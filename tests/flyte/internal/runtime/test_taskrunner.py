"""Tests for taskrunner behavior.

`run_task`: the `controller` argument is optional. Clustered/jobset tasks run with no controller
(they never enqueue subtasks); the only controller touchpoint on the leaf path is
`finalize_parent_action`, which must be skipped when there is none.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from flyte._internal.runtime.taskrunner import run_task


class _FakeTask:
    async def execute(self, **kwargs):
        return {"out": 1}


def test_run_task_without_controller_skips_finalize():
    tctx = SimpleNamespace(action="act-1")
    out, err = asyncio.run(run_task(tctx=tctx, controller=None, task=_FakeTask(), inputs={}))
    assert err is None
    assert out == {"out": 1}


def test_run_task_with_controller_finalizes():
    tctx = SimpleNamespace(action="act-1")
    controller = AsyncMock()
    _out, err = asyncio.run(run_task(tctx=tctx, controller=controller, task=_FakeTask(), inputs={}))
    assert err is None
    controller.finalize_parent_action.assert_awaited_once_with("act-1")
