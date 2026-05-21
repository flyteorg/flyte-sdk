"""Unit tests for remote → ActionTracker sync helpers."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

from flyte.cli._tui._tracker import ActionStatus, ActionTracker
from flyte.models import ActionPhase

from flyteplugins.remote_tui._sync import _phase_status, build_action_tree, load_run_into_tracker


@dataclass
class _FakeMeta:
    parent: str = ""
    group: str = ""


class _FakeAction:
    def __init__(
        self,
        name: str,
        *,
        parent: str = "",
        group: str = "",
        phase: ActionPhase = ActionPhase.SUCCEEDED,
        task_name: str | None = None,
        done: bool = True,
    ):
        self.name = name
        self.phase = phase
        self.task_name = task_name or name
        self._done = done
        self.pb2 = MagicMock()
        self.pb2.metadata = _FakeMeta(parent=parent, group=group)
        self.pb2.metadata.HasField = lambda field: field == "task" and task_name is not None
        self.pb2.metadata.task = MagicMock(short_name=None)
        self.pb2.status = MagicMock()
        self.pb2.status.attempts = 1
        self.pb2.status.phase = phase.to_protobuf_value() if hasattr(phase, "to_protobuf_value") else 5
        self.pb2.status.HasField = lambda _: done
        self.pb2.status.end_time = MagicMock()
        self.pb2.status.end_time.ToDatetime.return_value = __import__("datetime").datetime(
            2025, 1, 1, tzinfo=__import__("datetime").timezone.utc
        )
        self.pb2.attempts = []

    def done(self) -> bool:
        return self._done

    @property
    def start_time(self):
        return __import__("datetime").datetime(2025, 1, 1, tzinfo=__import__("datetime").timezone.utc)


def test_phase_status_maps_terminal_states():
    assert _phase_status(ActionPhase.SUCCEEDED) == ActionStatus.SUCCEEDED
    assert _phase_status(ActionPhase.FAILED) == ActionStatus.FAILED
    assert _phase_status(ActionPhase.RUNNING) == ActionStatus.RUNNING


def test_build_action_tree_parent_links():
    actions = [
        _FakeAction("root", parent=""),
        _FakeAction("child", parent="root"),
    ]
    parent_by_id, children = build_action_tree(actions)
    assert parent_by_id["child"] == "root"
    assert "child" in children["root"]


def test_load_run_into_tracker_creates_nodes():
    tracker = ActionTracker()
    actions = [
        _FakeAction("root", parent=""),
        _FakeAction("child", parent="root", phase=ActionPhase.RUNNING, done=False),
    ]
    load_run_into_tracker(tracker, actions, fetch_io=False)
    assert tracker.get_action("root") is not None
    assert tracker.get_action("child") is not None
    assert tracker.get_action("child").status == ActionStatus.RUNNING
