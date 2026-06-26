"""Unit tests for remote → ActionTracker sync helpers."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

from flyte.cli._tui._tracker import ActionStatus, ActionTracker
from flyte.models import ActionPhase

from flyteplugins.remote_tui._sync import (
    _condition_info_from_details,
    _is_condition_action,
    _phase_status,
    _prompt_type_str,
    build_action_tree,
    load_run_into_tracker,
)


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
    assert _phase_status(ActionPhase.PAUSED) == ActionStatus.PAUSED
    assert _phase_status("paused") == ActionStatus.PAUSED


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


class _FakeConditionAction(_FakeAction):
    def __init__(
        self,
        name: str,
        *,
        parent: str = "",
        phase: ActionPhase = ActionPhase.RUNNING,
        condition_name: str = "approve",
        done: bool = False,
    ):
        super().__init__(name, parent=parent, phase=phase, task_name=condition_name, done=done)
        self.pb2.metadata.HasField = lambda field: field == "condition"
        condition_meta = MagicMock()
        condition_meta.name = condition_name
        condition_meta.HasField = lambda field: False
        self.pb2.metadata.condition = condition_meta
        self.pb2.metadata.action_type = 3


def test_is_condition_action_detects_metadata():
    action = _FakeConditionAction("cond-1", parent="root")
    assert _is_condition_action(action) is True
    assert _is_condition_action(_FakeAction("task")) is False


def test_prompt_type_str_maps_markdown():
    from flyteidl2.workflow import run_definition_pb2

    assert _prompt_type_str(run_definition_pb2.CONDITION_PROMPT_TYPE_TEXT) == "text"
    assert _prompt_type_str(run_definition_pb2.CONDITION_PROMPT_TYPE_MARKDOWN) == "markdown"


def test_load_run_into_tracker_pending_condition():
    tracker = ActionTracker()
    actions = [
        _FakeAction("root", parent=""),
        _FakeConditionAction("cond-1", parent="root", phase=ActionPhase.RUNNING, done=False),
    ]
    load_run_into_tracker(tracker, actions, fetch_io=False)
    node = tracker.get_action("cond-1")
    assert node is not None
    assert node.status == ActionStatus.PAUSED
    pending = tracker.get_pending_condition("cond-1")
    assert pending is not None
    assert pending.condition_name == "approve"
    assert pending.data_type is str


def test_condition_info_from_details_reads_bool_type():
    from flyteidl2.core import types_pb2
    from flyteidl2.workflow import run_definition_pb2

    action = _FakeConditionAction("cond-1", parent="root", phase=ActionPhase.PAUSED, done=False)
    details = MagicMock()
    details.pb2 = run_definition_pb2.ActionDetails(
        condition=run_definition_pb2.ConditionAction(
            name="approve",
            type=types_pb2.LiteralType(simple=types_pb2.BOOLEAN),
            prompt="Approve?",
        ),
    )

    _, prompt, prompt_type, data_type, _ = _condition_info_from_details(action, details)
    assert prompt == "Approve?"
    assert prompt_type == "text"
    assert data_type is bool
