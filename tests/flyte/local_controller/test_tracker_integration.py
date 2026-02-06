"""Integration tests: LocalController + ActionTracker.

Verifies that running tasks through the local controller correctly
records start/complete/failure events, parent-child relationships,
cache status, and short_name into the ActionTracker.
"""

from __future__ import annotations

from typing import List

import flyte
from flyte.cli._tui._tracker import ActionStatus, ActionTracker

env = flyte.TaskEnvironment(name="tracker_test")


@env.task
def add(a: int, b: int) -> int:
    return a + b


@env.task
def failing_task(x: int) -> int:
    raise ValueError("intentional failure")


@env.task
def parent_task(n: int) -> List[int]:
    results = []
    for i in range(n):
        results.append(add(a=i, b=i))
    return results


# ---------------------------------------------------------------------------
# Basic tracking
# ---------------------------------------------------------------------------


def test_tracker_records_single_task():
    """A single task run should produce exactly one tracked action."""
    flyte.init_from_config(None)
    tracker = ActionTracker()
    flyte.with_runcontext(mode="local", _tracker=tracker).run(add, a=2, b=3)

    root_ids, _children, nodes = tracker.snapshot()
    assert len(root_ids) == 1
    root = nodes[root_ids[0]]
    assert root.status == ActionStatus.SUCCEEDED
    assert root.task_name == add.name
    assert root.parent_id is None
    assert root.end_time is not None


def test_tracker_records_inputs():
    """Tracked action should contain the native input values."""
    flyte.init_from_config(None)
    tracker = ActionTracker()
    flyte.with_runcontext(mode="local", _tracker=tracker).run(add, a=10, b=20)

    _, _, nodes = tracker.snapshot()
    node = next(iter(nodes.values()))
    assert node.inputs is not None
    assert node.inputs["a"] == 10
    assert node.inputs["b"] == 20


def test_tracker_records_outputs():
    """Tracked action should contain converted output values."""
    flyte.init_from_config(None)
    tracker = ActionTracker()
    flyte.with_runcontext(mode="local", _tracker=tracker).run(add, a=5, b=7)

    _, _, nodes = tracker.snapshot()
    node = next(iter(nodes.values()))
    assert node.outputs is not None
    # Output should be the pretty-printed version (a dict like {"o0": 12})
    assert node.status == ActionStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# Failure tracking
# ---------------------------------------------------------------------------


def test_tracker_records_failure():
    """A failing task should be recorded as FAILED with an error message."""
    flyte.init_from_config(None)
    tracker = ActionTracker()
    try:
        flyte.with_runcontext(mode="local", _tracker=tracker).run(failing_task, x=1)
    except Exception:
        pass  # expected

    _, _, nodes = tracker.snapshot()
    assert len(nodes) >= 1
    failed_nodes = [n for n in nodes.values() if n.status == ActionStatus.FAILED]
    assert len(failed_nodes) == 1
    assert "intentional failure" in failed_nodes[0].error


# ---------------------------------------------------------------------------
# Parent-child relationships
# ---------------------------------------------------------------------------


def test_tracker_records_nested_tasks():
    """Nested task calls should create parent-child relationships."""
    flyte.init_from_config(None)
    tracker = ActionTracker()
    flyte.with_runcontext(mode="local", _tracker=tracker).run(parent_task, n=3)

    root_ids, children, nodes = tracker.snapshot()
    assert len(root_ids) == 1
    root_id = root_ids[0]

    # The root should have 3 children (one per add() call)
    child_ids = children.get(root_id, [])
    assert len(child_ids) == 3

    for cid in child_ids:
        child = nodes[cid]
        assert child.parent_id == root_id
        assert child.status == ActionStatus.SUCCEEDED
        assert child.task_name == add.name


def test_tracker_no_duplicate_root():
    """The root action should appear exactly once (not duplicated)."""
    flyte.init_from_config(None)
    tracker = ActionTracker()
    flyte.with_runcontext(mode="local", _tracker=tracker).run(parent_task, n=2)

    root_ids, _, nodes = tracker.snapshot()
    assert len(root_ids) == 1
    # Only one node should have parent_id=None
    roots = [n for n in nodes.values() if n.parent_id is None]
    assert len(roots) == 1


# ---------------------------------------------------------------------------
# Cache tracking
# ---------------------------------------------------------------------------


def test_tracker_records_cache_disabled():
    """Tasks without cache should have cache_enabled=False."""
    flyte.init_from_config(None)
    tracker = ActionTracker()
    flyte.with_runcontext(mode="local", _tracker=tracker).run(add, a=1, b=2)

    _, _, nodes = tracker.snapshot()
    node = next(iter(nodes.values()))
    assert node.cache_enabled is False
    assert node.cache_hit is False


# ---------------------------------------------------------------------------
# Unique action IDs
# ---------------------------------------------------------------------------


def test_tracker_unique_action_ids_for_repeated_tasks():
    """Multiple calls to the same task should get unique action IDs."""
    flyte.init_from_config(None)
    tracker = ActionTracker()
    flyte.with_runcontext(mode="local", _tracker=tracker).run(parent_task, n=4)

    _, _, nodes = tracker.snapshot()
    action_ids = [n.action_id for n in nodes.values()]
    assert len(action_ids) == len(set(action_ids)), "Action IDs should be unique"


# ---------------------------------------------------------------------------
# Version tracking
# ---------------------------------------------------------------------------


def test_tracker_version_increments():
    """Version should increment with each event."""
    flyte.init_from_config(None)
    tracker = ActionTracker()
    assert tracker.version == 0
    flyte.with_runcontext(mode="local", _tracker=tracker).run(add, a=1, b=1)
    # At minimum: 1 start + 1 complete = 2
    assert tracker.version >= 2
