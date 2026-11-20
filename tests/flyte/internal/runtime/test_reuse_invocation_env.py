"""Tests for the per-invocation env vars of tasks running in reusable containers.

A reusable container's worker process runs many invocations, several at a time when its concurrency is
above one, and os.environ is shared by all of them. The worker therefore hands each invocation's env vars
to rusty.run_task, which keeps them on that invocation's TaskContext as invocation_env, and the running
action's names come from the TaskContext rather than from RUN_NAME and ACTION_NAME.
"""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import flyte
import flyte.report
from flyte._internal.runtime import rusty, taskrunner
from flyte.models import ActionID, RawDataPath, TaskContext


def _tctx(**kwargs) -> TaskContext:
    kwargs.setdefault("action", ActionID(name="a0", run_name="r0"))
    return TaskContext(
        version="v1",
        raw_data_path=RawDataPath(path="s3://bucket/raw"),
        output_path="s3://bucket/outputs",
        run_base_dir="s3://bucket",
        report=flyte.report.Report(name="a0"),
        **kwargs,
    )


def test_invocation_env_defaults_to_empty():
    assert dict(_tctx().invocation_env) == {}


def test_invocation_env_is_a_read_only_copy():
    source = {"RUN_NAME": "r1"}
    tctx = _tctx(invocation_env=source)
    source["RUN_NAME"] = "changed"

    assert tctx.invocation_env["RUN_NAME"] == "r1"
    with pytest.raises(TypeError):
        tctx.invocation_env["RUN_NAME"] = "x"  # type: ignore[index]


def test_invocation_env_survives_context_replace():
    # @trace and groups derive child contexts with replace(); the env must carry across.
    child = _tctx(invocation_env={"MY_VAR": "1"}).replace(action=ActionID(name="trace-1"))
    assert dict(child.invocation_env) == {"MY_VAR": "1"}


def test_invocation_env_is_left_out_of_the_repr():
    tctx = _tctx(invocation_env={"MY_SECRET": "hunter2"})
    assert "hunter2" not in repr(tctx)
    assert "invocation_env" not in repr(tctx)
    assert "action=" in repr(tctx)


def test_attempt_number_still_prefers_retry_attempt(monkeypatch):
    monkeypatch.setenv("FLYTE_ATTEMPT_NUMBER", "0")
    tctx = _tctx(retry_attempt=2, invocation_env={"FLYTE_ATTEMPT_NUMBER": "5"})
    assert tctx.attempt_number == 2


def test_null_context_has_no_invocation_env():
    assert not flyte.ctx()
    assert flyte.ctx().invocation_env is None


class _EnvReportingTask:
    """A task whose body records what flyte.ctx() and os.environ tell it about its invocation."""

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
        self._seen[self.name] = {
            "invocation_env": dict(flyte.ctx().invocation_env),
            "names": (flyte.ctx().action.run_name, flyte.ctx().action.name),
            "os_environ": os.environ.get("MY_RUN_VAR"),
        }
        return {}


def _patch_io(monkeypatch):
    monkeypatch.setattr(taskrunner, "convert_inputs_to_native", AsyncMock(return_value={}))
    monkeypatch.setattr(taskrunner, "convert_from_native_to_outputs", AsyncMock(return_value=SimpleNamespace()))
    monkeypatch.setattr(taskrunner, "upload_outputs", AsyncMock())


async def _rusty_run(task, run_name="r", env_vars=None):
    kwargs = {} if env_vars is None else {"env_vars": env_vars}
    await rusty.run_task(
        task=task,
        controller=None,
        org="org",
        project="p",
        domain="d",
        run_name=run_name,
        name=task.name,
        raw_data_path="s3://bucket/raw",
        output_path=f"s3://bucket/{task.name}/outputs",
        run_base_dir="s3://bucket",
        version="v1",
        **kwargs,
    )


def test_rusty_run_task_delivers_env_to_task_body_without_touching_os_environ(monkeypatch):
    monkeypatch.delenv("MY_RUN_VAR", raising=False)
    _patch_io(monkeypatch)
    seen: dict = {}
    env_vars = {"RUN_NAME": "r", "ACTION_NAME": "a1", "MY_RUN_VAR": "x", "EMPTY": ""}

    asyncio.run(_rusty_run(_EnvReportingTask("a1", seen), env_vars=env_vars))

    assert seen == {"a1": {"invocation_env": env_vars, "names": ("r", "a1"), "os_environ": None}}
    assert "MY_RUN_VAR" not in os.environ


def test_rusty_run_task_without_env_vars_leaves_invocation_env_empty(monkeypatch):
    # Workers that do not forward env_vars yet keep today's behavior: tasks read os.environ.
    monkeypatch.setenv("MY_RUN_VAR", "startup")
    _patch_io(monkeypatch)
    seen: dict = {}

    asyncio.run(_rusty_run(_EnvReportingTask("legacy", seen)))

    assert seen == {"legacy": {"invocation_env": {}, "names": ("r", "legacy"), "os_environ": "startup"}}


def test_concurrent_invocations_each_see_their_own_env_and_action(monkeypatch):
    # Two invocations share one event loop and one process; each must see only what came with it, while the
    # worker's env still names the action the replica started with.
    monkeypatch.setenv("RUN_NAME", "stale-run")
    monkeypatch.setenv("ACTION_NAME", "stale-action")
    monkeypatch.setenv("MY_RUN_VAR", "startup")
    _patch_io(monkeypatch)
    seen: dict = {}
    first_env = {"RUN_NAME": "run-1", "ACTION_NAME": "a1", "MY_RUN_VAR": "one"}
    second_env = {"RUN_NAME": "run-2", "ACTION_NAME": "a2", "MY_RUN_VAR": "two"}

    async def main():
        barrier: tuple[list, int, asyncio.Event] = ([], 2, asyncio.Event())
        await asyncio.wait_for(
            asyncio.gather(
                _rusty_run(_EnvReportingTask("a1", seen, barrier), run_name="run-1", env_vars=first_env),
                _rusty_run(_EnvReportingTask("a2", seen, barrier), run_name="run-2", env_vars=second_env),
            ),
            timeout=10,
        )

    asyncio.run(main())

    assert seen == {
        "a1": {"invocation_env": first_env, "names": ("run-1", "a1"), "os_environ": "startup"},
        "a2": {"invocation_env": second_env, "names": ("run-2", "a2"), "os_environ": "startup"},
    }
    assert os.environ["MY_RUN_VAR"] == "startup"
