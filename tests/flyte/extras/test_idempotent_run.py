"""Tests for `flyte.extras.idempotent_run`.

Idempotency lives on the `dedupe` run label, so these assert on the labels
passed to the run context — never on run names, which the control plane owns.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flyte.extras import DUPE_LABEL_KEY, DuplicateRun, blocking_run, idempotent_run


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


def _listall_returning(*runs):
    """Stand-in for the syncified `Run.listall`, whose `.aio()` is an async iterator."""

    async def aio(*args, **kwargs):
        for run in runs:
            yield run

    mock = MagicMock()
    mock.aio = aio
    return mock


def _runner_returning(run):
    runner = MagicMock()
    runner.run.aio = AsyncMock(return_value=run)
    return runner


@pytest.mark.asyncio
async def test_blocking_run_finds_a_live_run():
    live = _FakeRun(phase="RUNNING")
    with (
        patch("flyte.remote.Run.listall", _listall_returning(live)),
        patch("flyte.extras._idempotent_run._ensure_initialized"),
    ):
        assert await blocking_run.aio("k") is live


@pytest.mark.asyncio
async def test_blocking_run_ignores_retriable_phases():
    """A failed run is a retry opportunity, not a blocker."""
    for phase in ("FAILED", "ABORTED", "TIMED_OUT"):
        with (
            patch("flyte.remote.Run.listall", _listall_returning(_FakeRun(phase=phase))),
            patch("flyte.extras._idempotent_run._ensure_initialized"),
        ):
            assert await blocking_run.aio("k") is None


@pytest.mark.asyncio
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
        patch("flyte.extras._idempotent_run._ensure_initialized"),
    ):
        await blocking_run.aio("k")
    assert captured["with_labels"] == {DUPE_LABEL_KEY: "k"}


@pytest.mark.asyncio
async def test_raises_on_a_duplicate():
    live = _FakeRun(phase="RUNNING", name="somerun", url="http://run/somerun")
    with (
        patch("flyte.remote.Run.listall", _listall_returning(live)),
        patch("flyte.extras._idempotent_run._ensure_initialized"),
    ):
        with pytest.raises(DuplicateRun) as exc:
            await idempotent_run.aio(MagicMock(), key="k")
    assert exc.value.run_name == "somerun"
    assert exc.value.url == "http://run/somerun"


@pytest.mark.asyncio
async def test_labels_the_run_and_lets_the_platform_name_it():
    run = _FakeRun()
    runner = _runner_returning(run)
    with (
        patch("flyte.remote.Run.listall", _listall_returning()),
        patch("flyte.extras._idempotent_run._ensure_initialized"),
        patch("flyte.with_runcontext", return_value=runner) as with_runcontext,
    ):
        task = MagicMock()
        result = await idempotent_run.aio(task, key="k", x=1)

    assert result is run
    # Identity is the label; the run name is the control plane's to assign.
    with_runcontext.assert_called_once_with(labels={DUPE_LABEL_KEY: "k"})
    assert "name" not in with_runcontext.call_args.kwargs
    runner.run.aio.assert_awaited_once_with(task, x=1)


@pytest.mark.asyncio
async def test_forwards_copy_style():
    runner = _runner_returning(_FakeRun())
    with (
        patch("flyte.remote.Run.listall", _listall_returning()),
        patch("flyte.extras._idempotent_run._ensure_initialized"),
        patch("flyte.with_runcontext", return_value=runner) as with_runcontext,
    ):
        await idempotent_run.aio(MagicMock(), key="k", copy_style="all")
    with_runcontext.assert_called_once_with(labels={DUPE_LABEL_KEY: "k"}, copy_style="all")


def test_is_callable_synchronously_from_scripts():
    """The sync form works outside an event loop; handlers should use .aio()."""
    run = _FakeRun()
    with (
        patch("flyte.remote.Run.listall", _listall_returning()),
        patch("flyte.extras._idempotent_run._ensure_initialized"),
        patch("flyte.with_runcontext", return_value=_runner_returning(run)),
    ):
        assert idempotent_run(MagicMock(), key="k") is run


@pytest.mark.asyncio
async def test_launches_overlap_instead_of_serializing_on_the_event_loop():
    """Concurrent launches must overlap rather than queue behind one another.

    The stand-in runner blocks the calling thread in its synchronous form and
    yields in its async form — mirroring the real syncified runner, whose sync
    call form blocks on `future.result()`. An implementation reaching for the
    blocking form turns four 0.2s launches into ~0.8s of stalled event loop.
    """
    delay = 0.2

    def blocking_call(*args, **kwargs):
        time.sleep(delay)
        return _FakeRun()

    async def awaiting_call(*args, **kwargs):
        await asyncio.sleep(delay)
        return _FakeRun()

    runner = MagicMock()
    runner.run = MagicMock(side_effect=blocking_call)
    runner.run.aio = awaiting_call

    with (
        patch("flyte.remote.Run.listall", _listall_returning()),
        patch("flyte.extras._idempotent_run._ensure_initialized"),
        patch("flyte.with_runcontext", return_value=runner),
    ):
        started = time.perf_counter()
        results = await asyncio.gather(*(idempotent_run.aio(MagicMock(), key=f"k{i}") for i in range(4)))
        elapsed = time.perf_counter() - started

    assert len(results) == 4
    assert elapsed < delay * 2, f"launches serialized: {elapsed:.2f}s for 4 concurrent {delay}s launches"


def test_exposes_both_call_forms():
    assert hasattr(idempotent_run, "aio")
    assert hasattr(blocking_run, "aio")
