"""
Sub-action names must be stable across attempts of the same action when the parent runs in a
reusable container.

unionai-reuse creates ONE RemoteController per worker process and runs every action (and every
retry of an action) through it. The call sequence used to live on that controller, keyed by
``{action name}_{run name}`` -- identical for every attempt of an action -- so attempt N only started
from seq 1 if attempt N-1's ``finalize_parent_action`` cleared it, and did so before attempt N began.
The counters now live on the TaskContext, which is built fresh per attempt, so no cleanup is needed.

Each test drives attempts through the real ``taskrunner.run_task`` on one shared controller, with a
fresh TaskContext per attempt (as ``convert_and_run`` builds one); the task body makes three
identical ``foo()`` calls (same call key), exactly like:

    @env.task
    async def main():
        await foo()  # seq 1
        await foo()  # seq 2
        await foo()  # seq 3
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from flyte._internal.controllers.remote._controller import RemoteController
from flyte._internal.runtime.taskrunner import run_task
from flyte.models import ActionID, RawDataPath, TaskContext
from flyte.report import Report

CALL_KEY = "foo-task-identity:empty-inputs-hash"
N_CALLS = 3
EXPECTED = [1, 2, 3]

ACTION = ActionID(name="main-action", run_name="r1", project="p", domain="d", org="o")


def _controller() -> RemoteController:
    async def make_client():
        return AsyncMock()

    return RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)


def _names(seqs: list[int]) -> list[str]:
    return [ACTION.new_sub_action_from(task_call_seq=s, task_hash="foo", input_hash="h", group=None).name for s in seqs]


def _new_attempt_tctx(action: ActionID) -> TaskContext:
    return TaskContext(
        action=action,
        version="v1",
        raw_data_path=RawDataPath(path="/tmp/raw"),
        output_path="/tmp/out",
        run_base_dir="/tmp/base",
        report=Report(name="test"),
    )


class _Main:
    """Stand-in for the parent task: makes N identical foo() calls through the controller."""

    def __init__(self, controller: RemoteController, action: ActionID, *, fail: bool = False):
        self.controller = controller
        self.action = action
        self.tctx = _new_attempt_tctx(action)
        self.fail = fail
        self.seqs: list[int] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.hang = False

    async def call_foo(self) -> None:
        self.seqs.append(self.controller.generate_task_call_sequence(CALL_KEY, self.tctx))

    async def execute(self, **kwargs):
        for _ in range(N_CALLS):
            await self.call_foo()
        self.started.set()
        if self.hang:
            await self.release.wait()
        if self.fail:
            raise RuntimeError("fail this attempt to force a retry")
        return {}


async def _attempt(controller: RemoteController, task: _Main):
    return await run_task(tctx=task.tctx, controller=controller, task=task, inputs={})


@pytest.fixture
def finalize():
    with patch(
        "flyte._internal.controllers.remote._core.Controller._finalize_parent_action", new_callable=AsyncMock
    ) as m:
        yield m


@pytest.mark.asyncio
async def test_sequential_retry_after_user_error(finalize):
    """Attempt 0 raises, attempt 1 starts at seq 1 -> same names."""
    controller = _controller()

    a0 = _Main(controller, ACTION, fail=True)
    _, err = await _attempt(controller, a0)
    assert err is not None

    a1 = _Main(controller, ACTION)
    await _attempt(controller, a1)

    assert a0.seqs == EXPECTED
    assert a1.seqs == EXPECTED
    assert _names(a0.seqs) == _names(a1.seqs)


@pytest.mark.asyncio
async def test_back_to_back_runs_on_same_replica(finalize):
    """Control: a second run of the same task on the same replica gets the same sequence."""
    controller = _controller()
    r1 = _Main(controller, ACTION)
    r2 = _Main(controller, ActionID(name=ACTION.name, run_name="r2", project="p", domain="d", org="o"))
    await _attempt(controller, r1)
    await _attempt(controller, r2)
    assert r1.seqs == r2.seqs == EXPECTED


@pytest.mark.asyncio
async def test_concurrent_distinct_actions_on_same_replica(finalize):
    """Control: different actions sharing a replica (concurrency > 1) use different keys."""
    controller = _controller()
    a = _Main(controller, ACTION)
    b = _Main(controller, ActionID(name="other-action", run_name="r1", project="p", domain="d", org="o"))
    await asyncio.gather(_attempt(controller, a), _attempt(controller, b))
    assert a.seqs == b.seqs == EXPECTED


@pytest.mark.asyncio
async def test_retry_after_finalize_raises(finalize):
    """Attempt 0's finalize raises (e.g. informer stop error); attempt 1 must still start at seq 1."""
    controller = _controller()
    finalize.side_effect = [RuntimeError("informer stop failed"), None]

    a0 = _Main(controller, ACTION, fail=True)
    with pytest.raises(RuntimeError, match="informer stop failed"):
        await _attempt(controller, a0)

    a1 = _Main(controller, ACTION)
    await _attempt(controller, a1)

    assert a1.seqs == EXPECTED, f"retry minted new names: {_names(a1.seqs)} != {_names(a0.seqs)}"


@pytest.mark.asyncio
async def test_retry_after_attempt_cancelled_during_finalize(finalize):
    """Attempt 0 is cancelled (timeout/abort) and the cancel lands while finalize is awaiting the
    informer; attempt 1 on the same replica must still start at seq 1."""
    controller = _controller()
    in_finalize = asyncio.Event()

    async def slow_finalize(*args, **kwargs):
        in_finalize.set()
        await asyncio.Event().wait()

    finalize.side_effect = slow_finalize

    a0 = _Main(controller, ACTION)
    a0.hang = True
    t0 = asyncio.create_task(_attempt(controller, a0))
    await a0.started.wait()
    t0.cancel()  # timeout -> worker cancels the attempt; run_task's finally enters finalize
    await in_finalize.wait()
    t0.cancel()  # cancellation is delivered again while finalize is in flight
    with pytest.raises(asyncio.CancelledError):
        await t0

    finalize.side_effect = None
    a1 = _Main(controller, ACTION)
    await _attempt(controller, a1)

    assert a1.seqs == EXPECTED, f"retry minted new names: {_names(a1.seqs)} != {_names(a0.seqs)}"


@pytest.mark.asyncio
async def test_overlapping_attempts_on_same_replica(finalize):
    """Attempt 1 is scheduled on the replica while attempt 0 is still winding down (e.g. the platform
    gave up on attempt 0 but the worker hasn't finished cancelling it). Each attempt counts on its own
    context, so attempt 1 starts at 1 and attempt 0 finishing cannot reset attempt 1 mid-flight."""
    controller = _controller()

    a0 = _Main(controller, ACTION)
    a0.hang = True
    t0 = asyncio.create_task(_attempt(controller, a0))
    await a0.started.wait()

    a1 = _Main(controller, ACTION)
    a1.hang = True
    t1 = asyncio.create_task(_attempt(controller, a1))
    await a1.started.wait()

    a0.release.set()
    await t0  # attempt 0 finalizes while attempt 1 is still running
    await a1.call_foo()  # a 4th foo() in attempt 1 continues its own count
    a1.release.set()
    await t1

    assert a1.seqs == [*EXPECTED, 4], f"attempt 1 seqs {a1.seqs}"
