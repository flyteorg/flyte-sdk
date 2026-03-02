import pytest

from flyte._persistence._db import LocalDB
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


def test_explore_screen_imports():
    """Verify that the explore module can be imported without error."""


def test_runs_table_populate_empty():
    """RunsTable populates with no runs."""
    from flyte.cli._tui._explore import RunsTable

    RunStore.initialize_sync()
    RunsTable()
    # We can't call populate() outside of a textual app mount, but we can verify
    # the underlying data is empty
    runs = RunStore.list_runs_sync()
    assert len(runs) == 0


def test_runs_table_populate_with_data():
    """Verify RunStore data is available for the table."""
    RunStore.initialize_sync()
    RunStore.record_start_sync(run_name="run-1", action_name="a0", task_name="my_task")
    RunStore.record_complete_sync(run_name="run-1", action_name="a0")

    runs = RunStore.list_runs_sync()
    assert len(runs) == 1
    assert runs[0].run_name == "run-1"
    assert runs[0].status == "succeeded"


def test_run_detail_screen_tracker_reconstruction():
    """Verify the tracker is properly rebuilt from DB records."""
    from flyte.cli._tui._tracker import ActionStatus

    RunStore.initialize_sync()
    RunStore.record_start_sync(run_name="run-1", action_name="a0", task_name="root_task")
    RunStore.record_start_sync(run_name="run-1", action_name="a1", task_name="sub_task", parent_id="a0")
    RunStore.record_complete_sync(run_name="run-1", action_name="a1")
    RunStore.record_failure_sync(run_name="run-1", action_name="a0", error="boom")

    from flyte.cli._tui._explore import RunDetailScreen

    screen = RunDetailScreen("run-1")
    tracker = screen._tracker

    root = tracker.get_action("a0")
    assert root is not None
    assert root.task_name == "root_task"
    assert root.status == ActionStatus.FAILED

    sub = tracker.get_action("a1")
    assert sub is not None
    assert sub.task_name == "sub_task"
    assert sub.status == ActionStatus.SUCCEEDED


def test_run_detail_screen_tracker_extended_fields():
    """Verify extended fields (inputs, has_report, context, group, log_links) are rebuilt."""
    from flyte.cli._tui._tracker import ActionStatus

    RunStore.initialize_sync()
    RunStore.record_start_sync(
        run_name="run-ext",
        action_name="a0",
        task_name="ext_task",
        inputs={"x": 42},
        has_report=True,
        cache_enabled=True,
        cache_hit=True,
        disable_run_cache=False,
        context={"env": "staging"},
        group_name="grp1",
        log_links=[("logs", "http://example.com/logs")],
    )
    RunStore.record_complete_sync(run_name="run-ext", action_name="a0", outputs="done")

    from flyte.cli._tui._explore import RunDetailScreen

    screen = RunDetailScreen("run-ext")
    tracker = screen._tracker

    node = tracker.get_action("a0")
    assert node is not None
    assert node.inputs == {"x": 42}
    assert node.has_report is True
    assert node.cache_enabled is True
    assert node.cache_hit is True
    assert node.disable_run_cache is False
    assert node.context == {"env": "staging"}
    assert node.group == "grp1"
    assert node.log_links == [["logs", "http://example.com/logs"]]
    assert node.status == ActionStatus.SUCCEEDED


def test_run_detail_screen_tracker_disable_run_cache():
    """Verify disable_run_cache is rebuilt from DB when loading RunDetailScreen."""

    RunStore.initialize_sync()
    RunStore.record_start_sync(
        run_name="run-disabled-cache",
        action_name="a0",
        task_name="cached_task",
        cache_enabled=True,
        cache_hit=True,
        disable_run_cache=True,
    )
    RunStore.record_complete_sync(run_name="run-disabled-cache", action_name="a0", outputs="ok")

    from flyte.cli._tui._explore import RunDetailScreen

    screen = RunDetailScreen("run-disabled-cache")
    tracker = screen._tracker

    node = tracker.get_action("a0")
    assert node is not None
    assert node.disable_run_cache is True


def test_run_detail_screen_rebuilds_attempt_history():
    RunStore.initialize_sync()
    RunStore.record_start_sync(run_name="run-attempts", action_name="a0", task_name="retry_task")
    RunStore.record_attempt_start_sync(run_name="run-attempts", action_name="a0", attempt_num=1)
    RunStore.record_attempt_failure_sync(run_name="run-attempts", action_name="a0", attempt_num=1, error="boom")
    RunStore.record_attempt_start_sync(run_name="run-attempts", action_name="a0", attempt_num=2)
    RunStore.record_attempt_complete_sync(run_name="run-attempts", action_name="a0", attempt_num=2, outputs="ok")
    RunStore.record_complete_sync(run_name="run-attempts", action_name="a0", outputs="ok")

    from flyte.cli._tui._explore import RunDetailScreen

    screen = RunDetailScreen("run-attempts")
    node = screen._tracker.get_action("a0")
    assert node is not None
    assert node.attempt_count == 2
    assert len(node.attempts) == 2


def test_fmt_duration():
    from flyte.cli._tui._explore import _fmt_duration

    assert _fmt_duration(None, None) == ""
    assert _fmt_duration(100.0, None) == "running..."
    assert _fmt_duration(100.0, 103.5) == "3.50s"


def test_fmt_time():
    from flyte.cli._tui._explore import _fmt_time

    assert _fmt_time(None) == ""
    result = _fmt_time(1700000000.0)
    assert "2023" in result  # Nov 14, 2023


def test_fmt_attempts():
    from flyte.cli._tui._explore import _fmt_attempts

    assert _fmt_attempts(1) == "x1"
    assert _fmt_attempts(3) == "x3"


def test_next_attempt_num():
    from flyte.cli._tui._app import _next_attempt_num

    attempts = [1, 2, 3]
    assert _next_attempt_num(None, attempts, +1) == 3
    assert _next_attempt_num(3, attempts, -1) == 2
    assert _next_attempt_num(2, attempts, +1) == 3
    assert _next_attempt_num(1, attempts, -1) == 1


def test_cache_icon_without_disable_run_cache():
    """_cache_icon returns cache status when disable_run_cache is False."""
    from flyte.cli._tui._app import _cache_icon
    from flyte.cli._tui._tracker import ActionNode, ActionStatus

    # cache_hit -> " $"
    node_hit = ActionNode(
        action_id="a1",
        task_name="t",
        parent_id=None,
        status=ActionStatus.SUCCEEDED,
        cache_enabled=True,
        cache_hit=True,
        disable_run_cache=False,
    )
    assert _cache_icon(node_hit) == " $"

    # cache_enabled but miss -> " ~"
    node_miss = ActionNode(
        action_id="a1",
        task_name="t",
        parent_id=None,
        status=ActionStatus.SUCCEEDED,
        cache_enabled=True,
        cache_hit=False,
        disable_run_cache=False,
    )
    assert _cache_icon(node_miss) == " ~"

    # no cache -> ""
    node_no_cache = ActionNode(
        action_id="a1",
        task_name="t",
        parent_id=None,
        status=ActionStatus.SUCCEEDED,
        cache_enabled=False,
        cache_hit=False,
        disable_run_cache=False,
    )
    assert _cache_icon(node_no_cache) == ""


def test_cache_icon_with_disable_run_cache():
    """_cache_icon returns '-' when disable_run_cache is True (run-level override)."""
    from flyte.cli._tui._app import _cache_icon
    from flyte.cli._tui._tracker import ActionNode, ActionStatus

    # Even with cache_hit=True, disable_run_cache overrides to "-"
    node = ActionNode(
        action_id="a1",
        task_name="t",
        parent_id=None,
        status=ActionStatus.SUCCEEDED,
        cache_enabled=True,
        cache_hit=True,
        disable_run_cache=True,
    )
    assert _cache_icon(node) == "-"


def test_compute_runs_table_column_widths():
    from flyte.cli._tui._explore import _compute_runs_table_column_widths

    widths = _compute_runs_table_column_widths(120)
    assert len(widths) == 7
    assert sum(widths) >= 112  # total_width - borders/separators
    assert widths[4] >= 19  # start time column stays readable
