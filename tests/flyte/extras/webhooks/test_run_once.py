"""Tests for `flyte.extras.webhooks.run_once`.

Deduplication lives on the `dedupe` run label, so these assert on the labels
passed to the run context — never on run names, which the control plane owns.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flyte.extras.webhooks import DUPE_LABEL_KEY, RunOnceResult, blocking_run, run_once


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
        patch("flyte.extras.webhooks._run_once._ensure_initialized"),
    ):
        assert await blocking_run.aio("k") is live


@pytest.mark.asyncio
async def test_blocking_run_ignores_retriable_phases():
    """A failed run is a retry opportunity, not a blocker."""
    for phase in ("FAILED", "ABORTED", "TIMED_OUT"):
        with (
            patch("flyte.remote.Run.listall", _listall_returning(_FakeRun(phase=phase))),
            patch("flyte.extras.webhooks._run_once._ensure_initialized"),
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
        patch("flyte.extras.webhooks._run_once._ensure_initialized"),
    ):
        await blocking_run.aio("k")
    assert captured["with_labels"] == {DUPE_LABEL_KEY: "k"}


@pytest.mark.asyncio
async def test_returns_the_existing_run_on_a_duplicate():
    live = _FakeRun(phase="RUNNING", name="somerun", url="http://run/somerun")
    with (
        patch("flyte.remote.Run.listall", _listall_returning(live)),
        patch("flyte.extras.webhooks._run_once._ensure_initialized"),
        patch("flyte.with_runcontext") as with_runcontext,
    ):
        result = await run_once.aio(MagicMock(), key="k")

    assert result == RunOnceResult(run=live, created=False)
    assert result.run.url == "http://run/somerun"
    # The whole point: the duplicate path must not reach the control plane.
    with_runcontext.assert_not_called()


@pytest.mark.asyncio
async def test_the_result_unpacks_as_a_tuple():
    """`run, created = await run_once.aio(...)` is a documented call form."""
    live = _FakeRun(phase="RUNNING")
    with (
        patch("flyte.remote.Run.listall", _listall_returning(live)),
        patch("flyte.extras.webhooks._run_once._ensure_initialized"),
    ):
        run, created = await run_once.aio(MagicMock(), key="k")

    assert (run, created) == (live, False)


@pytest.mark.asyncio
async def test_labels_the_run_and_lets_the_platform_name_it():
    run = _FakeRun()
    runner = _runner_returning(run)
    with (
        patch("flyte.remote.Run.listall", _listall_returning()),
        patch("flyte.extras.webhooks._run_once._ensure_initialized"),
        patch("flyte.with_runcontext", return_value=runner) as with_runcontext,
    ):
        task = MagicMock()
        result = await run_once.aio(task, key="k", x=1)

    assert result == RunOnceResult(run=run, created=True)
    # Identity is the label; the run name is the control plane's to assign.
    with_runcontext.assert_called_once_with(labels={DUPE_LABEL_KEY: "k"})
    assert "name" not in with_runcontext.call_args.kwargs
    runner.run.aio.assert_awaited_once_with(task, x=1)


@pytest.mark.asyncio
async def test_forwards_copy_style():
    runner = _runner_returning(_FakeRun())
    with (
        patch("flyte.remote.Run.listall", _listall_returning()),
        patch("flyte.extras.webhooks._run_once._ensure_initialized"),
        patch("flyte.with_runcontext", return_value=runner) as with_runcontext,
    ):
        await run_once.aio(MagicMock(), key="k", copy_style="all")
    with_runcontext.assert_called_once_with(labels={DUPE_LABEL_KEY: "k"}, copy_style="all")


def test_is_callable_synchronously_from_scripts():
    """The sync form works outside an event loop; handlers should use .aio()."""
    run = _FakeRun()
    with (
        patch("flyte.remote.Run.listall", _listall_returning()),
        patch("flyte.extras.webhooks._run_once._ensure_initialized"),
        patch("flyte.with_runcontext", return_value=_runner_returning(run)),
    ):
        assert run_once(MagicMock(), key="k") == RunOnceResult(run=run, created=True)


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
        patch("flyte.extras.webhooks._run_once._ensure_initialized"),
        patch("flyte.with_runcontext", return_value=runner),
    ):
        started = time.perf_counter()
        results = await asyncio.gather(*(run_once.aio(MagicMock(), key=f"k{i}") for i in range(4)))
        elapsed = time.perf_counter() - started

    assert len(results) == 4
    assert elapsed < delay * 2, f"launches serialized: {elapsed:.2f}s for 4 concurrent {delay}s launches"


def test_exposes_both_call_forms():
    assert hasattr(run_once, "aio")
    assert hasattr(blocking_run, "aio")


@pytest.mark.asyncio
async def test_runcontext_kwargs_are_forwarded():
    """Anything else the run needs — queue, env_vars, service_account, ..."""
    runner = _runner_returning(_FakeRun())
    with (
        patch("flyte.remote.Run.listall", _listall_returning()),
        patch("flyte.extras.webhooks._run_once._ensure_initialized"),
        patch("flyte.with_runcontext", return_value=runner) as with_runcontext,
    ):
        await run_once.aio(
            MagicMock(),
            key="k",
            runcontext_kwargs={"queue": "webhooks", "env_vars": {"A": "1"}, "interruptible": True},
        )
    kwargs = with_runcontext.call_args.kwargs
    assert kwargs["queue"] == "webhooks"
    assert kwargs["env_vars"] == {"A": "1"}
    assert kwargs["interruptible"] is True


@pytest.mark.asyncio
async def test_extra_labels_merge_with_the_dedupe_label():
    """Callers can label their runs without displacing what makes them idempotent."""
    runner = _runner_returning(_FakeRun())
    with (
        patch("flyte.remote.Run.listall", _listall_returning()),
        patch("flyte.extras.webhooks._run_once._ensure_initialized"),
        patch("flyte.with_runcontext", return_value=runner) as with_runcontext,
    ):
        await run_once.aio(MagicMock(), key="k", runcontext_kwargs={"labels": {"team": "platform"}})
    assert with_runcontext.call_args.kwargs["labels"] == {"team": "platform", DUPE_LABEL_KEY: "k"}


@pytest.mark.asyncio
async def test_setting_the_dedupe_label_yourself_is_refused():
    """Silently honouring it would be indistinguishable from disabling idempotency."""
    with (
        patch("flyte.remote.Run.listall", _listall_returning()),
        patch("flyte.extras.webhooks._run_once._ensure_initialized"),
    ):
        with pytest.raises(ValueError, match="break idempotency"):
            await run_once.aio(MagicMock(), key="k", runcontext_kwargs={"labels": {DUPE_LABEL_KEY: "something-else"}})


@pytest.mark.asyncio
async def test_restating_the_same_dedupe_label_is_harmless():
    runner = _runner_returning(_FakeRun())
    with (
        patch("flyte.remote.Run.listall", _listall_returning()),
        patch("flyte.extras.webhooks._run_once._ensure_initialized"),
        patch("flyte.with_runcontext", return_value=runner) as with_runcontext,
    ):
        await run_once.aio(MagicMock(), key="k", runcontext_kwargs={"labels": {DUPE_LABEL_KEY: "k"}})
    assert with_runcontext.call_args.kwargs["labels"] == {DUPE_LABEL_KEY: "k"}


@pytest.mark.asyncio
async def test_copy_style_given_twice_is_refused():
    with (
        patch("flyte.remote.Run.listall", _listall_returning()),
        patch("flyte.extras.webhooks._run_once._ensure_initialized"),
    ):
        with pytest.raises(ValueError, match="not both"):
            await run_once.aio(MagicMock(), key="k", copy_style="all", runcontext_kwargs={"copy_style": "none"})


@pytest.mark.asyncio
async def test_copy_style_can_come_through_runcontext_kwargs():
    runner = _runner_returning(_FakeRun())
    with (
        patch("flyte.remote.Run.listall", _listall_returning()),
        patch("flyte.extras.webhooks._run_once._ensure_initialized"),
        patch("flyte.with_runcontext", return_value=runner) as with_runcontext,
    ):
        await run_once.aio(MagicMock(), key="k", runcontext_kwargs={"copy_style": "all"})
    assert with_runcontext.call_args.kwargs["copy_style"] == "all"


@pytest.mark.asyncio
async def test_the_callers_dict_is_not_mutated():
    """A handler may reuse one dict across events."""
    shared = {"labels": {"team": "platform"}, "queue": "webhooks"}
    runner = _runner_returning(_FakeRun())
    with (
        patch("flyte.remote.Run.listall", _listall_returning()),
        patch("flyte.extras.webhooks._run_once._ensure_initialized"),
        patch("flyte.with_runcontext", return_value=runner),
    ):
        await run_once.aio(MagicMock(), key="k", runcontext_kwargs=shared)
    assert shared == {"labels": {"team": "platform"}, "queue": "webhooks"}
