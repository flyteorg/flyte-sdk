"""Tests for the attempt number seen by tasks running in reusable containers.

A reusable container's worker process starts once and then runs many executions, retries included,
so its FLYTE_ATTEMPT_NUMBER environment variable keeps whatever value the replica started with. The
attempt therefore has to travel with each invocation (rusty.run_task) into the TaskContext, and
TaskContext.attempt_number must prefer it over the process environment.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import flyte
from flyte._internal.runtime import rusty, taskrunner
from flyte.models import ActionID, RawDataPath, TaskContext


def _tctx(**kwargs) -> TaskContext:
    return TaskContext(
        action=ActionID(name="a0"),
        version="v1",
        raw_data_path=RawDataPath(path="s3://bucket/raw"),
        output_path="s3://bucket/outputs",
        run_base_dir="s3://bucket",
        report=flyte.report.Report(name="a0"),
        **kwargs,
    )


def test_attempt_number_prefers_per_invocation_value_over_env(monkeypatch):
    # The worker's environment still says 0, as it did in the real run that hit this bug.
    monkeypatch.setenv("FLYTE_ATTEMPT_NUMBER", "0")
    assert _tctx(retry_attempt=1).attempt_number == 1


def test_attempt_number_falls_back_to_env_when_not_delivered(monkeypatch):
    monkeypatch.setenv("FLYTE_ATTEMPT_NUMBER", "2")
    assert _tctx().attempt_number == 2


def test_attempt_number_defaults_to_zero(monkeypatch):
    monkeypatch.delenv("FLYTE_ATTEMPT_NUMBER", raising=False)
    assert _tctx().attempt_number == 0


def test_retry_attempt_survives_context_replace(monkeypatch):
    # @trace and groups derive child contexts with replace(); the attempt must carry across.
    monkeypatch.setenv("FLYTE_ATTEMPT_NUMBER", "0")
    child = _tctx(retry_attempt=3).replace(action=ActionID(name="trace-1"))
    assert child.attempt_number == 3


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, None), (0, 0), (2, 2), ("1", 1), (" 4 ", 4), ("", None), ("x", None), (-1, None), (True, None)],
)
def test_parse_attempt_number(raw, expected):
    assert rusty._parse_attempt_number(raw) == expected


class _AttemptReportingTask:
    """A task whose body records the attempt number it observes through flyte.ctx()."""

    report = False
    native_interface = SimpleNamespace()

    def __init__(self, name: str, seen: dict, barrier: tuple[list, int, asyncio.Event] | None = None):
        self.name = name
        self._seen = seen
        self._barrier = barrier

    async def execute(self, **kwargs):
        if self._barrier is not None:
            # Hold every invocation open until all of them are inside their task body, so they overlap.
            started, expected, all_started = self._barrier
            started.append(self.name)
            if len(started) == expected:
                all_started.set()
            await all_started.wait()
        self._seen[self.name] = flyte.ctx().attempt_number
        return {}


def _patch_io(monkeypatch):
    monkeypatch.setattr(taskrunner, "convert_inputs_to_native", AsyncMock(return_value={}))
    monkeypatch.setattr(taskrunner, "convert_from_native_to_outputs", AsyncMock(return_value=SimpleNamespace()))
    monkeypatch.setattr(taskrunner, "upload_outputs", AsyncMock())


async def _rusty_run(task, attempt_number=None):
    kwargs = {} if attempt_number is None else {"attempt_number": attempt_number}
    await rusty.run_task(
        task=task,
        controller=None,
        org="org",
        project="p",
        domain="d",
        run_name="r",
        name=task.name,
        raw_data_path="s3://bucket/raw",
        output_path=f"s3://bucket/{task.name}/outputs",
        run_base_dir="s3://bucket",
        version="v1",
        **kwargs,
    )


def test_rusty_run_task_delivers_attempt_to_task_body(monkeypatch):
    monkeypatch.setenv("FLYTE_ATTEMPT_NUMBER", "0")
    _patch_io(monkeypatch)
    seen: dict = {}

    asyncio.run(_rusty_run(_AttemptReportingTask("retry", seen), attempt_number="1"))

    assert seen == {"retry": 1}


def test_rusty_run_task_without_attempt_keeps_env_fallback(monkeypatch):
    # Workers that do not forward the attempt yet keep today's behavior.
    monkeypatch.setenv("FLYTE_ATTEMPT_NUMBER", "0")
    _patch_io(monkeypatch)
    seen: dict = {}

    asyncio.run(_rusty_run(_AttemptReportingTask("legacy", seen)))

    assert seen == {"legacy": 0}


def test_concurrent_invocations_each_see_their_own_attempt(monkeypatch):
    # A replica with concurrency > 1 runs several invocations on one event loop in one process, so the
    # attempt cannot live in os.environ; each invocation's context has to hold its own value.
    monkeypatch.setenv("FLYTE_ATTEMPT_NUMBER", "0")
    _patch_io(monkeypatch)
    seen: dict = {}

    async def main():
        barrier: tuple[list, int, asyncio.Event] = ([], 3, asyncio.Event())
        await asyncio.wait_for(
            asyncio.gather(
                *(_rusty_run(_AttemptReportingTask(f"t{n}", seen, barrier), attempt_number=n) for n in range(3))
            ),
            timeout=10,
        )

    asyncio.run(main())

    assert seen == {"t0": 0, "t1": 1, "t2": 2}
