import threading
from unittest.mock import MagicMock, patch

import pytest

from flyte._persistence._db import LocalDB
from flyte._persistence._recorder import RunRecorder
from flyte._persistence._run_store import RunStore


@pytest.fixture(autouse=True)
def _reset_db(tmp_path, monkeypatch):
    """Reset LocalDB state before each test and point it at a temp directory."""
    LocalDB._conn = None
    LocalDB._conn_sync = None
    LocalDB._initialized = False
    db_path = str(tmp_path / "cache.db")
    monkeypatch.setattr(LocalDB, "_get_db_path", staticmethod(lambda: db_path))
    yield
    if LocalDB._conn_sync:
        LocalDB._conn_sync.close()
    LocalDB._conn = None
    LocalDB._conn_sync = None
    LocalDB._initialized = False


def _make_tracker():
    """Create a mock tracker matching ActionTracker's interface."""
    tracker = MagicMock()
    tracker.get_action.return_value = None
    return tracker


class TestIsActive:
    def test_neither_backend(self):
        r = RunRecorder()
        assert r.is_active is False

    def test_tracker_only(self):
        r = RunRecorder(tracker=_make_tracker())
        assert r.is_active is True

    def test_persist_only(self):
        r = RunRecorder(persist=True, run_name="run-1")
        assert r.is_active is True

    def test_both_backends(self):
        r = RunRecorder(tracker=_make_tracker(), persist=True, run_name="run-1")
        assert r.is_active is True

    def test_persist_without_run_name_is_inactive(self):
        r = RunRecorder(persist=True, run_name=None)
        assert r.is_active is False


class TestGetAction:
    def test_delegates_to_tracker(self):
        tracker = _make_tracker()
        sentinel = object()
        tracker.get_action.return_value = sentinel
        r = RunRecorder(tracker=tracker)
        assert r.get_action("a1") is sentinel
        tracker.get_action.assert_called_once_with("a1")

    def test_returns_none_without_tracker(self):
        r = RunRecorder()
        assert r.get_action("a1") is None


class TestBothBackends:
    def test_record_start_calls_both(self):
        tracker = _make_tracker()
        RunStore.initialize_sync()
        r = RunRecorder(tracker=tracker, persist=True, run_name="run-1")

        r.record_start(
            action_id="a1",
            task_name="my_task",
            parent_id=None,
            short_name="mt",
            inputs={"x": 1},
            output_path="/tmp/out",
            has_report=True,
            cache_enabled=True,
            cache_hit=False,
            context={"env": "test"},
            group="grp",
            log_links=[("log", "http://localhost")],
        )

        # Tracker called with original parent_id (None)
        tracker.record_start.assert_called_once()
        call_kwargs = tracker.record_start.call_args.kwargs
        assert call_kwargs["action_id"] == "a1"
        assert call_kwargs["parent_id"] is None
        assert call_kwargs["group"] == "grp"

        # Persistence: parent_id=None -> "a0"
        action = RunStore.get_action_sync("run-1", "a1")
        assert action is not None
        assert action.parent_id == "a0"
        assert action.task_name == "my_task"
        assert action.group_name == "grp"

    def test_record_start_with_disable_run_cache(self):
        """Verify disable_run_cache is passed to both tracker and persistence."""
        tracker = _make_tracker()
        RunStore.initialize_sync()
        r = RunRecorder(tracker=tracker, persist=True, run_name="run-1")

        r.record_start(
            action_id="a1",
            task_name="my_task",
            parent_id=None,
            short_name="mt",
            disable_run_cache=True,
            cache_enabled=True,
            cache_hit=True,  # Would show hit, but disable_run_cache overrides
        )

        call_kwargs = tracker.record_start.call_args.kwargs
        assert call_kwargs["disable_run_cache"] is True

        action = RunStore.get_action_sync("run-1", "a1")
        assert action is not None
        assert action.disable_run_cache is True

    def test_record_complete_calls_both(self):
        tracker = _make_tracker()
        RunStore.initialize_sync()
        r = RunRecorder(tracker=tracker, persist=True, run_name="run-1")

        RunStore.record_start_sync(run_name="run-1", action_name="a1", task_name="t")
        r.record_complete(action_id="a1", outputs="some result")

        tracker.record_complete.assert_called_once()
        assert tracker.record_complete.call_args.kwargs["action_id"] == "a1"
        # Strings pass through _to_display unchanged
        assert tracker.record_complete.call_args.kwargs["outputs"] == "some result"

        action = RunStore.get_action_sync("run-1", "a1")
        assert action is not None
        assert action.status == "succeeded"
        assert action.outputs is not None

    def test_record_failure_calls_both(self):
        tracker = _make_tracker()
        RunStore.initialize_sync()
        r = RunRecorder(tracker=tracker, persist=True, run_name="run-1")

        RunStore.record_start_sync(run_name="run-1", action_name="a1", task_name="t")
        r.record_failure(action_id="a1", error="boom")

        tracker.record_failure.assert_called_once_with(action_id="a1", error="boom")

        action = RunStore.get_action_sync("run-1", "a1")
        assert action is not None
        assert action.status == "failed"
        assert action.error == "boom"

    def test_record_attempt_calls_both(self):
        tracker = _make_tracker()
        RunStore.initialize_sync()
        r = RunRecorder(tracker=tracker, persist=True, run_name="run-1")

        RunStore.record_start_sync(run_name="run-1", action_name="a1", task_name="t")
        r.record_attempt_start(action_id="a1", attempt_num=1)
        r.record_attempt_failure(action_id="a1", attempt_num=1, error="retry me")
        r.record_attempt_start(action_id="a1", attempt_num=2)
        r.record_attempt_complete(action_id="a1", attempt_num=2, outputs="ok")

        tracker.record_attempt_start.assert_any_call(action_id="a1", attempt_num=1)
        tracker.record_attempt_failure.assert_any_call(action_id="a1", attempt_num=1, error="retry me")
        tracker.record_attempt_complete.assert_any_call(action_id="a1", attempt_num=2, outputs="ok")

        action = RunStore.get_action_sync("run-1", "a1")
        assert action is not None
        assert action.attempt_count == 2
        assert action.attempts_json is not None


class TestTrackerOnly:
    def test_no_persistence_calls(self):
        tracker = _make_tracker()
        r = RunRecorder(tracker=tracker)

        r.record_start(action_id="a1", task_name="t")
        r.record_complete(action_id="a1")

        tracker.record_start.assert_called_once()
        tracker.record_complete.assert_called_once()
        # No DB initialized, so RunStore calls would fail — absence of error proves no call.

    def test_record_failure_tracker_only(self):
        tracker = _make_tracker()
        r = RunRecorder(tracker=tracker)

        r.record_failure(action_id="a1", error="err")
        tracker.record_failure.assert_called_once_with(action_id="a1", error="err")


class TestPersistOnly:
    def test_no_tracker_calls(self):
        RunStore.initialize_sync()
        r = RunRecorder(persist=True, run_name="run-1")

        r.record_start(action_id="a1", task_name="t", parent_id="a0")
        r.record_complete(action_id="a1")

        action = RunStore.get_action_sync("run-1", "a1")
        assert action is not None
        assert action.status == "succeeded"


class TestNoOpRecorder:
    def test_no_op_does_nothing(self):
        r = RunRecorder()
        # Should not raise
        r.record_start(action_id="a1", task_name="t")
        r.record_complete(action_id="a1")
        r.record_failure(action_id="a2", error="err")
        r.record_root_start(task_name="t")
        r.record_root_complete()
        r.record_root_failure(error="err")


class TestRootActions:
    def test_record_root_start(self):
        RunStore.initialize_sync()
        r = RunRecorder(persist=True, run_name="run-1")

        r.record_root_start(task_name="root_task")

        action = RunStore.get_action_sync("run-1", "a0")
        assert action is not None
        assert action.task_name == "root_task"
        assert action.status == "running"

    def test_record_root_complete(self):
        RunStore.initialize_sync()
        r = RunRecorder(persist=True, run_name="run-1")

        RunStore.record_start_sync(run_name="run-1", action_name="a0", task_name="t")
        r.record_root_complete()

        action = RunStore.get_action_sync("run-1", "a0")
        assert action is not None
        assert action.status == "succeeded"

    def test_record_root_failure(self):
        RunStore.initialize_sync()
        r = RunRecorder(persist=True, run_name="run-1")

        RunStore.record_start_sync(run_name="run-1", action_name="a0", task_name="t")
        r.record_root_failure(error="root error")

        action = RunStore.get_action_sync("run-1", "a0")
        assert action is not None
        assert action.status == "failed"
        assert action.error == "root error"

    def test_root_methods_skip_tracker(self):
        tracker = _make_tracker()
        RunStore.initialize_sync()
        r = RunRecorder(tracker=tracker, persist=True, run_name="run-1")

        r.record_root_start(task_name="t")
        r.record_root_complete()

        # Tracker should not have been called for root actions
        tracker.record_start.assert_not_called()
        tracker.record_complete.assert_not_called()


class TestOutputConversion:
    def test_string_output_passes_through(self):
        """String outputs bypass literal_string_repr and pass through as-is."""
        RunStore.initialize_sync()
        tracker = _make_tracker()
        r = RunRecorder(tracker=tracker, persist=True, run_name="run-1")

        RunStore.record_start_sync(run_name="run-1", action_name="a1", task_name="t")
        r.record_complete(action_id="a1", outputs="hello")

        # Tracker gets the string directly
        tracker.record_complete.assert_called_once()
        assert tracker.record_complete.call_args.kwargs["outputs"] == "hello"

        # RunStore gets repr("hello") == "'hello'"
        action = RunStore.get_action_sync("run-1", "a1")
        assert action is not None
        assert action.outputs == "'hello'"

    def test_non_string_output_uses_literal_string_repr(self):
        """Non-string outputs are converted via literal_string_repr."""
        RunStore.initialize_sync()
        tracker = _make_tracker()
        r = RunRecorder(tracker=tracker, persist=True, run_name="run-1")

        RunStore.record_start_sync(run_name="run-1", action_name="a1", task_name="t")

        sentinel = {"x": 42}
        with patch("flyte.types._string_literals.literal_string_repr", return_value=sentinel) as mock_lsr:
            r.record_complete(action_id="a1", outputs=object())

        mock_lsr.assert_called_once()
        # Both get the same converted value
        assert tracker.record_complete.call_args.kwargs["outputs"] is sentinel
        action = RunStore.get_action_sync("run-1", "a1")
        assert action is not None
        assert action.outputs == repr(sentinel)

    def test_output_falls_back_to_repr(self):
        """When literal_string_repr fails, falls back to repr()."""
        RunStore.initialize_sync()
        tracker = _make_tracker()
        r = RunRecorder(tracker=tracker, persist=True, run_name="run-1")

        RunStore.record_start_sync(run_name="run-1", action_name="a1", task_name="t")

        with patch("flyte.types._string_literals.literal_string_repr", side_effect=Exception("boom")):
            r.record_complete(action_id="a1", outputs=42)

        # Both get repr(42) == "42" as the display string
        assert tracker.record_complete.call_args.kwargs["outputs"] == "42"
        action = RunStore.get_action_sync("run-1", "a1")
        assert action is not None
        assert action.outputs == "'42'"

    def test_output_none(self):
        RunStore.initialize_sync()
        tracker = _make_tracker()
        r = RunRecorder(tracker=tracker, persist=True, run_name="run-1")

        RunStore.record_start_sync(run_name="run-1", action_name="a1", task_name="t")
        r.record_complete(action_id="a1", outputs=None)

        assert tracker.record_complete.call_args.kwargs["outputs"] is None
        action = RunStore.get_action_sync("run-1", "a1")
        assert action is not None
        assert action.outputs is None


class TestInitializePersistence:
    def test_initializes_db(self):
        assert not LocalDB._initialized
        RunRecorder.initialize_persistence()
        assert LocalDB._initialized


class TestThreadSafety:
    def test_concurrent_record_calls(self):
        RunStore.initialize_sync()
        tracker = _make_tracker()
        r = RunRecorder(tracker=tracker, persist=True, run_name="run-1")

        errors = []

        def worker(i):
            try:
                r.record_start(action_id=f"a{i}", task_name=f"task-{i}", parent_id="a0")
                r.record_complete(action_id=f"a{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert tracker.record_start.call_count == 20
        assert tracker.record_complete.call_count == 20

        actions = RunStore.list_actions_for_run_sync("run-1")
        assert len(actions) == 20
