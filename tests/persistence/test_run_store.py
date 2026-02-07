import threading
import time

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


def test_record_start_and_list():
    RunStore.initialize_sync()
    RunStore.record_start_sync(
        run_name="run-1",
        action_name="a0",
        task_name="my_task",
        parent_id=None,
    )
    runs = RunStore.list_runs_sync()
    assert len(runs) == 1
    assert runs[0].run_name == "run-1"
    assert runs[0].action_name == "a0"
    assert runs[0].task_name == "my_task"
    assert runs[0].status == "running"
    assert runs[0].start_time is not None


def test_record_complete():
    RunStore.initialize_sync()
    RunStore.record_start_sync(run_name="run-1", action_name="a0", task_name="my_task")
    RunStore.record_complete_sync(run_name="run-1", action_name="a0", outputs="result")

    runs = RunStore.list_runs_sync()
    assert runs[0].status == "succeeded"
    assert runs[0].outputs == "result"
    assert runs[0].end_time is not None


def test_record_failure():
    RunStore.initialize_sync()
    RunStore.record_start_sync(run_name="run-1", action_name="a0", task_name="my_task")
    RunStore.record_failure_sync(run_name="run-1", action_name="a0", error="boom")

    runs = RunStore.list_runs_sync()
    assert runs[0].status == "failed"
    assert runs[0].error == "boom"
    assert runs[0].end_time is not None


def test_list_actions_for_run():
    RunStore.initialize_sync()
    RunStore.record_start_sync(run_name="run-1", action_name="a0", task_name="root_task")
    RunStore.record_start_sync(run_name="run-1", action_name="a1", task_name="sub_task", parent_id="a0")
    RunStore.record_complete_sync(run_name="run-1", action_name="a1")
    RunStore.record_complete_sync(run_name="run-1", action_name="a0")

    actions = RunStore.list_actions_for_run_sync("run-1")
    assert len(actions) == 2
    assert actions[0].action_name == "a0"
    assert actions[1].action_name == "a1"


def test_get_action():
    RunStore.initialize_sync()
    RunStore.record_start_sync(run_name="run-1", action_name="a0", task_name="my_task")

    action = RunStore.get_action_sync("run-1", "a0")
    assert action is not None
    assert action.task_name == "my_task"

    missing = RunStore.get_action_sync("run-1", "nonexistent")
    assert missing is None


def test_delete_run():
    RunStore.initialize_sync()
    RunStore.record_start_sync(run_name="run-1", action_name="a0", task_name="t1")
    RunStore.record_start_sync(run_name="run-1", action_name="a1", task_name="t2", parent_id="a0")
    RunStore.record_start_sync(run_name="run-2", action_name="a0", task_name="t3")

    RunStore.delete_run_sync("run-1")

    runs = RunStore.list_runs_sync()
    assert len(runs) == 1
    assert runs[0].run_name == "run-2"

    # Sub-actions should also be deleted
    actions = RunStore.list_actions_for_run_sync("run-1")
    assert len(actions) == 0


def test_delete_runs_batch():
    RunStore.initialize_sync()
    RunStore.record_start_sync(run_name="run-1", action_name="a0", task_name="t1")
    RunStore.record_start_sync(run_name="run-2", action_name="a0", task_name="t2")
    RunStore.record_start_sync(run_name="run-3", action_name="a0", task_name="t3")

    RunStore.delete_runs_sync(["run-1", "run-2"])

    runs = RunStore.list_runs_sync()
    assert len(runs) == 1
    assert runs[0].run_name == "run-3"


def test_clear():
    RunStore.initialize_sync()
    RunStore.record_start_sync(run_name="run-1", action_name="a0", task_name="t1")
    RunStore.record_start_sync(run_name="run-2", action_name="a0", task_name="t2")

    RunStore.clear_sync()

    runs = RunStore.list_runs_sync()
    assert len(runs) == 0


def test_cache_fields():
    RunStore.initialize_sync()
    RunStore.record_start_sync(
        run_name="run-1",
        action_name="a0",
        task_name="t1",
        cache_enabled=True,
        cache_hit=True,
    )
    action = RunStore.get_action_sync("run-1", "a0")
    assert action is not None
    assert action.cache_enabled is True
    assert action.cache_hit is True


def test_inputs_stored_as_json():
    RunStore.initialize_sync()
    RunStore.record_start_sync(
        run_name="run-1",
        action_name="a0",
        task_name="t1",
        inputs={"x": 42, "y": "hello"},
    )
    action = RunStore.get_action_sync("run-1", "a0")
    assert action is not None
    assert '"x": 42' in action.inputs
    assert '"y": "hello"' in action.inputs


def test_thread_safety():
    RunStore.initialize_sync()
    errors = []

    def worker(i):
        try:
            RunStore.record_start_sync(run_name=f"run-{i}", action_name="a0", task_name=f"task-{i}")
            RunStore.record_complete_sync(run_name=f"run-{i}", action_name="a0")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    runs = RunStore.list_runs_sync()
    assert len(runs) == 20


def test_list_runs_ordered_by_start_time_desc():
    RunStore.initialize_sync()
    RunStore.record_start_sync(run_name="run-old", action_name="a0", task_name="t1")
    time.sleep(0.01)
    RunStore.record_start_sync(run_name="run-new", action_name="a0", task_name="t2")

    runs = RunStore.list_runs_sync()
    assert runs[0].run_name == "run-new"
    assert runs[1].run_name == "run-old"


def test_extended_fields():
    """Verify has_report, context, group_name, and log_links are persisted."""
    RunStore.initialize_sync()
    RunStore.record_start_sync(
        run_name="run-1",
        action_name="a0",
        task_name="t1",
        has_report=True,
        cache_enabled=True,
        cache_hit=False,
        context={"env": "prod"},
        group_name="my-group",
        log_links=[("console", "http://localhost:8080")],
        inputs={"x": 1},
    )

    action = RunStore.get_action_sync("run-1", "a0")
    assert action is not None
    assert action.has_report is True
    assert action.cache_enabled is True
    assert action.cache_hit is False
    assert '"env"' in action.context
    assert action.group_name == "my-group"
    assert "console" in action.log_links
    assert '"x": 1' in action.inputs
