"""Tests for dispatch/idempotency helpers.

Idempotency lives on the `dedupe` run label, so these assert on the labels
passed to the run context — never on run names, which the control plane owns.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flyteplugins.linear._dispatch import (
    DUPE_LABEL_KEY,
    DuplicateRun,
    blocking_run,
    launch_task,
)


def _listall_returning(*runs):
    """A stand-in for the syncified `Run.listall`, whose `.aio()` is an async iterator."""

    async def aio(*args, **kwargs):
        for run in runs:
            yield run

    mock = MagicMock()
    mock.aio = aio
    return mock


class _FakeRun:
    """A stand-in for a run.

    Deliberately not a `MagicMock`: MagicMock defines `__aiter__`, and syncify
    treats any result with `__aiter__` as an async iterator — so a mocked run
    comes back from the synchronous call form as a generator.
    """

    def __init__(self, phase: str = "SUCCEEDED", name: str = "r1", url: str = "http://run"):
        self.phase = phase
        self.name = name
        self.url = url


def _run(phase: str, name: str = "r1", url: str = "http://run"):
    return _FakeRun(phase=phase, name=name, url=url)


def _runner_returning(run):
    """A stand-in for `flyte.with_runcontext(...)`, whose `.run.aio()` is awaitable."""
    runner = MagicMock()
    runner.run.aio = AsyncMock(return_value=run)
    return runner


async def test_blocking_run_finds_live_run():
    live = _run("RUNNING")
    with (
        patch("flyte.remote.Run.listall", _listall_returning(live)),
        patch("flyteplugins.linear._dispatch._ensure_flyte_initialized"),
    ):
        assert await blocking_run.aio("k") is live


async def test_blocking_run_ignores_retriable_phases():
    """A failed run is a retry opportunity, not a blocker."""
    for phase in ("FAILED", "ABORTED", "TIMED_OUT"):
        with (
            patch("flyte.remote.Run.listall", _listall_returning(_run(phase))),
            patch("flyteplugins.linear._dispatch._ensure_flyte_initialized"),
        ):
            assert await blocking_run.aio("k") is None


async def test_blocking_run_queries_by_label():
    captured = {}

    async def aio(*args, **kwargs):
        captured.update(kwargs)
        return
        yield  # pragma: no cover - makes this an async generator

    listall = MagicMock()
    listall.aio = aio
    with (
        patch("flyte.remote.Run.listall", listall),
        patch("flyteplugins.linear._dispatch._ensure_flyte_initialized"),
    ):
        await blocking_run.aio("k")
    assert captured["with_labels"] == {DUPE_LABEL_KEY: "k"}


async def test_launch_task_raises_on_duplicate():
    live = _run("RUNNING", name="somerun", url="http://run/somerun")
    with (
        patch("flyte.remote.Run.listall", _listall_returning(live)),
        patch("flyteplugins.linear._dispatch._ensure_flyte_initialized"),
    ):
        with pytest.raises(DuplicateRun) as exc:
            await launch_task.aio(MagicMock(), key="k")
    assert exc.value.run_name == "somerun"


async def test_launch_task_labels_the_run_and_lets_the_platform_name_it():
    run = _FakeRun()
    runner = _runner_returning(run)
    with (
        patch("flyte.remote.Run.listall", _listall_returning()),
        patch("flyteplugins.linear._dispatch._ensure_flyte_initialized"),
        patch("flyte.with_runcontext", return_value=runner) as with_runcontext,
    ):
        task = MagicMock()
        result = await launch_task.aio(task, key="k", some_input="abc", number=1)

    assert result is run
    # Identity is the label; the run name is the control plane's to assign.
    with_runcontext.assert_called_once_with(labels={DUPE_LABEL_KEY: "k"})
    assert "name" not in with_runcontext.call_args.kwargs
    runner.run.aio.assert_awaited_once_with(task, some_input="abc", number=1)


async def test_launch_task_forwards_copy_style():
    runner = _runner_returning(_FakeRun())
    with (
        patch("flyte.remote.Run.listall", _listall_returning()),
        patch("flyteplugins.linear._dispatch._ensure_flyte_initialized"),
        patch("flyte.with_runcontext", return_value=runner) as with_runcontext,
    ):
        await launch_task.aio(MagicMock(), key="k", copy_style="all")
    with_runcontext.assert_called_once_with(labels={DUPE_LABEL_KEY: "k"}, copy_style="all")


async def test_a_user_supplied_key_sets_the_idempotency_scope():
    """The key is just a string, so callers can choose any scope they want."""
    runner = _runner_returning(_FakeRun())
    with (
        patch("flyte.remote.Run.listall", _listall_returning()),
        patch("flyteplugins.linear._dispatch._ensure_flyte_initialized"),
        patch("flyte.with_runcontext", return_value=runner) as with_runcontext,
    ):
        await launch_task.aio(MagicMock(), key="my-own-scope")
    with_runcontext.assert_called_once_with(labels={DUPE_LABEL_KEY: "my-own-scope"})


def test_launch_task_is_callable_synchronously_from_scripts():
    """The sync form still works outside an event loop; handlers should use .aio()."""
    run = _FakeRun()
    runner = _runner_returning(run)
    with (
        patch("flyte.remote.Run.listall", _listall_returning()),
        patch("flyteplugins.linear._dispatch._ensure_flyte_initialized"),
        patch("flyte.with_runcontext", return_value=runner),
    ):
        assert launch_task(MagicMock(), key="k") is run


def test_launch_task_exposes_an_async_form():
    assert hasattr(launch_task, "aio")
    assert hasattr(blocking_run, "aio")


async def test_launches_overlap_instead_of_serializing_on_the_event_loop():
    """Concurrent launches must overlap rather than queue behind one another.

    The stand-in runner blocks the calling thread in its synchronous form and
    yields in its async form — mirroring the real syncified runner, whose sync
    call form blocks on `future.result()`. So an implementation that reaches
    for the blocking form turns four 0.2s launches into ~0.8s of stalled event
    loop, while awaiting `.aio()` finishes all four in ~0.2s.
    """
    import asyncio
    import time

    delay = 0.2

    def blocking_run_call(*args, **kwargs):
        time.sleep(delay)
        return _FakeRun()

    async def awaiting_run_call(*args, **kwargs):
        await asyncio.sleep(delay)
        return _FakeRun()

    runner = MagicMock()
    runner.run = MagicMock(side_effect=blocking_run_call)
    runner.run.aio = awaiting_run_call

    with (
        patch("flyte.remote.Run.listall", _listall_returning()),
        patch("flyteplugins.linear._dispatch._ensure_flyte_initialized"),
        patch("flyte.with_runcontext", return_value=runner),
    ):
        started = time.perf_counter()
        results = await asyncio.gather(*(launch_task.aio(MagicMock(), key=f"k{i}") for i in range(4)))
        elapsed = time.perf_counter() - started

    assert len(results) == 4
    assert elapsed < delay * 2, f"launches serialized: {elapsed:.2f}s for 4 concurrent {delay}s launches"
