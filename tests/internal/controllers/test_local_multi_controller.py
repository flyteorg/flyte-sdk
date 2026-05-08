"""Tests for the multiprocessing local controller (`local-multi` mode).

These tests actually spawn worker processes via ``ProcessPoolExecutor``,
so they are slower than the in-process LocalController tests. Tasks must
be defined at module level so cloudpickle can resolve them in the worker.
"""

from __future__ import annotations

import asyncio
import os

import pytest

import flyte

# Module-level so cloudpickle can find them in the worker.
env = flyte.TaskEnvironment("test_local_multi")


@env.task
async def echo_pid(x: int) -> dict:
    return {"pid": os.getpid(), "x": x}


@env.task
async def add(a: int, b: int) -> int:
    return a + b


@env.task
async def parent_fanout() -> list:
    rs = await asyncio.gather(*[echo_pid(i) for i in range(3)])
    return rs


@env.task
async def parent_sequential() -> int:
    a = await add(1, 2)
    b = await add(a, 4)
    return b


@env.task
async def always_fail() -> int:
    raise RuntimeError("always_fail boom")


@env.task
async def _hard_exit_leaf() -> int:
    # Simulate OOM-kill / segfault: bypass Python exception machinery
    # and terminate the worker process abruptly. Must be called via a
    # parent task so this runs in a worker, not the test process itself.
    os._exit(137)
    return 0  # unreachable


@env.task
async def hard_exit_via_parent() -> int:
    return await _hard_exit_leaf()


def test_subprocess_executes_in_different_pid():
    """A submitted sub-task must run in a process that is not the parent."""
    parent_pid = os.getpid()
    result = flyte.with_runcontext(mode="local-multi").run(parent_fanout)
    outputs = result.outputs().o0
    assert len(outputs) == 3
    pids = {r["pid"] for r in outputs}
    assert parent_pid not in pids, f"sub-tasks ran in parent pid {parent_pid}, got {pids}"


def test_fanout_runs_in_distinct_workers():
    """asyncio.gather of N sub-tasks should fan out across N distinct workers."""
    result = flyte.with_runcontext(mode="local-multi").run(parent_fanout)
    outputs = result.outputs().o0
    pids = {r["pid"] for r in outputs}
    assert len(pids) >= 2, f"expected fan-out across distinct workers, got {pids}"


def test_sequential_subtasks_return_correct_values():
    """Sub-task outputs must marshal back across the process boundary."""
    result = flyte.with_runcontext(mode="local-multi").run(parent_sequential)
    assert result.outputs().o0 == 7  # (1+2)+4


def test_failure_propagates_to_parent():
    """An unrecoverable exception in the worker must reach the caller."""
    with pytest.raises(Exception):
        flyte.with_runcontext(mode="local-multi").run(always_fail)


def test_retry_then_success(tmp_path):
    """A recoverable failure on the first attempt should retry and succeed.

    Uses a sentinel file because the attempt counter must be visible across
    the parent and worker process boundaries.
    """
    sentinel = tmp_path / "attempts.txt"
    sentinel_path = str(sentinel)

    retry_env = flyte.TaskEnvironment("test_local_multi_retry")

    @retry_env.task(retries=1)
    async def retrying(x: int) -> int:
        # Cross-process sentinel: blocking open is intentional in this test.
        try:
            with open(sentinel_path) as f:  # noqa: ASYNC230
                n = int(f.read().strip())
        except FileNotFoundError:
            n = 0
        n += 1
        with open(sentinel_path, "w") as f:  # noqa: ASYNC230
            f.write(str(n))
        if n == 1:
            raise ValueError("retrying: first attempt fails")
        return x * 2

    result = flyte.with_runcontext(mode="local-multi").run(retrying, 5)
    assert result.outputs().o0 == 10


def test_worker_hard_exit_propagates_clean_error():
    """A worker that calls os._exit (OOM, segfault, etc.) should surface a
    clear error to the caller, not an internal concurrent.futures stack
    trace. The leaf is wrapped in a parent task so the death happens in
    a worker process — not in the test process."""
    with pytest.raises(Exception) as ei:
        flyte.with_runcontext(mode="local-multi").run(hard_exit_via_parent)
    msg = str(ei.value).lower()
    assert "worker" in msg or "process" in msg, f"unhelpful error message: {ei.value!r}"


def test_worker_hard_exit_recovers_with_retry(tmp_path):
    """If retries are configured, the controller should reset its broken
    pool after a worker death and let a fresh worker run the retry.

    The retrying task must be a SUB-task (called from a parent task) so
    that it runs in a worker process — only sub-tasks dispatch to the
    pool; the root task body always runs in-process."""
    sentinel = tmp_path / "die_once.txt"
    sentinel_path = str(sentinel)

    retry_env = flyte.TaskEnvironment("test_local_multi_die_retry")

    @retry_env.task(retries=1)
    async def maybe_die_leaf() -> int:
        try:
            with open(sentinel_path) as f:  # noqa: ASYNC230
                n = int(f.read().strip())
        except FileNotFoundError:
            n = 0
        n += 1
        with open(sentinel_path, "w") as f:  # noqa: ASYNC230
            f.write(str(n))
        if n == 1:
            os._exit(137)
        return 99

    @retry_env.task
    async def maybe_die_parent() -> int:
        return await maybe_die_leaf()

    result = flyte.with_runcontext(mode="local-multi").run(maybe_die_parent)
    assert result.outputs().o0 == 99


def test_stop_terminates_all_workers():
    """controller.stop() must kill any worker subprocesses it started so
    the parent process can exit promptly."""
    import time

    from flyte._internal.controllers import _ControllerState, create_controller
    from flyte._internal.controllers.multi import LocalMultiController

    # Reset global controller state so this test gets a fresh instance.
    _ControllerState.controller = None

    ctrl = create_controller("local-multi")
    assert isinstance(ctrl, LocalMultiController)

    pool = ctrl._get_pool()
    # Force the pool to actually start workers by submitting trivial work.
    futures = [pool.submit(os.getpid) for _ in range(2)]
    pids = [f.result(timeout=15) for f in futures]
    assert all(pid > 0 for pid in pids)

    worker_procs = list(pool._processes.values())
    assert worker_procs, "expected workers to be running"

    asyncio.new_event_loop().run_until_complete(ctrl.stop())

    # Give the OS a moment to reap.
    time.sleep(0.5)
    leaked = [p for p in worker_procs if p.is_alive()]
    assert not leaked, f"stop() left {len(leaked)} worker(s) alive: {[p.pid for p in leaked]}"
    # Reset state so the next test gets a fresh controller.
    _ControllerState.controller = None
