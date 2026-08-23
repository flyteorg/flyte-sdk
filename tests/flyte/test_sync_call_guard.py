"""Guard: blocking invocation of a sync task from an async task body must raise.

A sync-style call (``child(...)``) inside an ``async def`` task blocks the event loop that
drives the task body via ``concurrent.futures.Future.result(None)``. That loop also services
``controller.watch_for_errors``, so if the controller/informer dies the process can never
observe the failure and hangs forever. ``__call__`` therefore rejects the call and points the
user at ``await child.aio(...)``.

The guard simply checks for a running loop in the calling thread: a *sync* task body runs as
plain sync code on a dedicated thread with no loop (``run_sync_in_thread``), so the
long-supported sync-parent -> sync-child blocking call must keep working.
"""

import concurrent.futures
from unittest.mock import MagicMock, patch

import pytest

import flyte
import flyte.report
from flyte._context import internal_ctx
from flyte.errors import SyncTaskCallInAsyncContextError
from flyte.models import ActionID, RawDataPath, TaskContext

env = flyte.TaskEnvironment(name="sync-call-guard-test")


@env.task
def sync_child() -> str:
    return "done"


@env.task
async def async_parent_blocking() -> str:
    return sync_child()


@env.task
async def async_parent_aio() -> str:
    return await sync_child.aio()


@env.task
def sync_parent() -> str:
    return sync_child()


def _make_tctx() -> TaskContext:
    return TaskContext(
        action=ActionID(name="parent"),
        raw_data_path=RawDataPath(path="test"),
        output_path="/tmp",
        version="v1",
        run_base_dir="/run_base",
        report=flyte.report.Report(name="test_report"),
    )


def _resolved_future(value: str) -> concurrent.futures.Future:
    fut: concurrent.futures.Future = concurrent.futures.Future()
    fut.set_result(value)
    return fut


@pytest.mark.asyncio
async def test_blocking_sync_call_from_async_task_raises():
    controller = MagicMock()
    with (
        internal_ctx().replace_task_context(_make_tctx()),
        patch("flyte._internal.controllers.get_controller", return_value=controller),
    ):
        with pytest.raises(SyncTaskCallInAsyncContextError, match=r"await sync_child\.aio"):
            await async_parent_blocking.execute()
    controller.submit_sync.assert_not_called()


@pytest.mark.asyncio
async def test_aio_call_from_async_task_is_allowed():
    controller = MagicMock()
    controller.submit_sync.return_value = _resolved_future("done")
    with (
        internal_ctx().replace_task_context(_make_tctx()),
        patch("flyte._internal.controllers.get_controller", return_value=controller),
    ):
        assert await async_parent_aio.execute() == "done"
    controller.submit_sync.assert_called_once()


def test_blocking_sync_call_from_async_task_raises_in_local_run():
    flyte.init()
    # The local controller re-raises the failure as RuntimeUserError; match on the guidance text.
    with pytest.raises(flyte.errors.RuntimeUserError, match=r"await sync_child\.aio"):
        flyte.with_runcontext(mode="local").run(async_parent_blocking)


@pytest.mark.asyncio
async def test_blocking_sync_call_from_sync_task_is_allowed():
    # The sync parent body runs as plain sync code on a dedicated thread (no running loop);
    # blocking there is safe and must keep working.
    controller = MagicMock()
    controller.submit_sync.return_value = _resolved_future("done")
    with (
        internal_ctx().replace_task_context(_make_tctx()),
        patch("flyte._internal.controllers.get_controller", return_value=controller),
    ):
        assert await sync_parent.execute() == "done"
    controller.submit_sync.assert_called_once()
