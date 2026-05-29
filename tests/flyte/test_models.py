from __future__ import annotations

import pytest

import flyte.report
from flyte.models import ActionID, PathRewrite, RawDataPath, TaskContext


def test_task_context_replace_preserves_parent_action():
    """`_trace.py` swaps `tctx.action` per trace scope via `.replace(action=...)`.
    `parent_action` must survive that swap — otherwise trace-bookkeeping would
    nest under the previous trace's pseudo-action instead of the real container."""
    container = ActionID(name="real_container_action", run_name="r", project="p", domain="d", org="o")
    trace_pseudo = ActionID(name="trace_pseudo_action", run_name="r", project="p", domain="d", org="o")

    tctx = TaskContext(
        action=container,
        parent_action=container,
        raw_data_path=RawDataPath(path="x"),
        output_path="/tmp",
        version="v1",
        run_base_dir="/tmp/base",
        report=flyte.report.Report(name="r"),
    )

    swapped = tctx.replace(action=trace_pseudo)
    assert swapped.action == trace_pseudo
    assert swapped.parent_action == container

    # Nesting another trace on top: action gets swapped again, parent_action still pinned.
    deeper_pseudo = ActionID(name="deeper_pseudo", run_name="r", project="p", domain="d", org="o")
    swapped2 = swapped.replace(action=deeper_pseudo)
    assert swapped2.action == deeper_pseudo
    assert swapped2.parent_action == container


def test_task_context_parent_action_defaults_to_none():
    """Legacy/test constructors that omit parent_action get None — the controller
    falls back to action.name in that case (preserves backwards compatibility)."""
    tctx = TaskContext(
        action=ActionID(name="a"),
        raw_data_path=RawDataPath(path="x"),
        output_path="/tmp",
        version="v1",
        run_base_dir="/tmp/base",
        report=flyte.report.Report(name="r"),
    )
    assert tctx.parent_action is None


def test_path_rewrite():
    with pytest.raises(ValueError):
        PathRewrite.from_str("s3://old_prefix:/tmp/new_prefix")

    with pytest.raises(ValueError):
        PathRewrite.from_str("s3://old_prefix-/tmp/new_prefix")

    with pytest.raises(ValueError):
        PathRewrite.from_str("s3://old_prefix/tmp/new_prefix")

    with pytest.raises(ValueError):
        PathRewrite.from_str("s3://old_prefix- >/tmp/new_prefix")

    pr = PathRewrite.from_str("s3://old_prefix/->/tmp/new_prefix/")
    assert pr.old_prefix == "s3://old_prefix/"
    assert pr.new_prefix == "/tmp/new_prefix/"
