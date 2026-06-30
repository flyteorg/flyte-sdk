"""Local controller: trace records nest under task_action, mirroring the remote record_trace fix.

When @trace swaps tctx.action for a per-trace pseudo-action, tctx.task_action stays pinned to the
real running task. The local controller must record trace lineage under task_action so local TUI /
run-store nesting matches the remote action tree.
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

import pytest

import flyte
import flyte.report
from flyte._context import internal_ctx
from flyte._internal.controllers._local_controller import LocalController
from flyte.models import ActionID, CodeBundle, NativeInterface, RawDataPath, TaskContext

this_dir_str = str(pathlib.Path(__file__).parent.absolute())


def _make_tctx(**overrides) -> TaskContext:
    defaults = {
        "action": ActionID(name="parent"),
        "raw_data_path": RawDataPath(path="test"),
        "output_path": "/tmp",
        "version": "v1",
        "run_base_dir": "/run_base",
        "report": flyte.report.Report(name="test_report"),
        "code_bundle": CodeBundle(computed_version="vcode-bundle", destination=this_dir_str, tgz="dummy.tgz"),
    }
    defaults.update(overrides)
    return TaskContext(**defaults)


async def _fn() -> int:
    return 1


@pytest.mark.asyncio
async def test_local_trace_reparents_to_task_action_inside_trace():
    """Inside a @trace scope (action != task_action), the recorded trace parents under task_action."""
    await flyte.init.aio()
    controller = LocalController()
    recorder = MagicMock()
    recorder.is_active = True
    controller.set_recorder(recorder)

    interface = NativeInterface(inputs={}, outputs={"result": int})
    tctx = _make_tctx(
        action=ActionID(name="outer_trace_action"),
        task_action=ActionID(name="real_container_action"),
    )

    ctx = internal_ctx()
    with ctx.replace_task_context(tctx):
        await controller.get_action_outputs(interface, _fn)

    recorder.record_start.assert_called_once()
    assert recorder.record_start.call_args.kwargs["parent_id"] == "real_container_action"


@pytest.mark.asyncio
async def test_local_trace_parent_is_action_for_regular_task():
    """Outside a trace, task_action defaults to action, so the recorded parent is tctx.action.name."""
    await flyte.init.aio()
    controller = LocalController()
    recorder = MagicMock()
    recorder.is_active = True
    controller.set_recorder(recorder)

    interface = NativeInterface(inputs={}, outputs={"result": int})
    tctx = _make_tctx(action=ActionID(name="real_task"))  # task_action omitted → defaults to action

    ctx = internal_ctx()
    with ctx.replace_task_context(tctx):
        await controller.get_action_outputs(interface, _fn)

    recorder.record_start.assert_called_once()
    assert recorder.record_start.call_args.kwargs["parent_id"] == "real_task"
