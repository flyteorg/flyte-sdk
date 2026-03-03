"""Tests for flyte.cli._tui._tracker — ActionTracker edge cases and conversions."""

from __future__ import annotations

import threading
import time

from flyte.cli._tui._tracker import ActionNode, ActionStatus, ActionTracker, _safe_json

# ---------------------------------------------------------------------------
# _safe_json
# ---------------------------------------------------------------------------


class TestSafeJson:
    def test_none(self):
        assert _safe_json(None) is None

    def test_primitives(self):
        assert _safe_json(True) is True
        assert _safe_json(42) == 42
        assert _safe_json(3.14) == 3.14
        assert _safe_json("hello") == "hello"

    def test_dict(self):
        result = _safe_json({"a": 1, "b": "two"})
        assert result == {"a": 1, "b": "two"}

    def test_nested_dict(self):
        result = _safe_json({"a": {"b": [1, 2, 3]}})
        assert result == {"a": {"b": [1, 2, 3]}}

    def test_list(self):
        assert _safe_json([1, "two", 3.0]) == [1, "two", 3.0]

    def test_tuple_becomes_list(self):
        assert _safe_json((1, 2)) == [1, 2]

    def test_non_serializable_falls_back_to_repr(self):
        obj = object()
        result = _safe_json(obj)
        assert isinstance(result, str)
        assert "object" in result

    def test_dict_with_non_serializable_value(self):
        obj = object()
        result = _safe_json({"key": obj})
        assert isinstance(result["key"], str)
        assert "object" in result["key"]

    def test_dict_with_non_string_key(self):
        result = _safe_json({1: "one", 2: "two"})
        assert result == {"1": "one", "2": "two"}

    def test_list_with_non_serializable(self):
        obj = object()
        result = _safe_json([1, obj, "three"])
        assert result[0] == 1
        assert isinstance(result[1], str)
        assert result[2] == "three"

    def test_bool_not_treated_as_int(self):
        # bool is a subclass of int; make sure it stays bool
        assert _safe_json(True) is True
        assert _safe_json(False) is False


# ---------------------------------------------------------------------------
# ActionNode
# ---------------------------------------------------------------------------


class TestActionNode:
    def test_defaults(self):
        node = ActionNode(
            action_id="a1",
            task_name="my_task",
            parent_id=None,
            status=ActionStatus.RUNNING,
        )
        assert node.short_name is None
        assert node.inputs is None
        assert node.outputs is None
        assert node.error is None
        assert node.output_path is None
        assert not node.has_report
        assert not node.cache_enabled
        assert not node.cache_hit
        assert not node.disable_run_cache
        assert node.context is None
        assert node.end_time is None
        assert isinstance(node.start_time, float)

    def test_short_name(self):
        node = ActionNode(
            action_id="a1",
            task_name="module.my_task",
            parent_id=None,
            status=ActionStatus.RUNNING,
            short_name="my_task",
        )
        assert node.short_name == "my_task"


# ---------------------------------------------------------------------------
# ActionTracker — record_start
# ---------------------------------------------------------------------------


class TestTrackerRecordStart:
    def test_basic_root(self):
        t = ActionTracker()
        t.record_start(action_id="a1", task_name="task1")
        node = t.get_action("a1")
        assert node is not None
        assert node.task_name == "task1"
        assert node.parent_id is None
        assert node.status == ActionStatus.RUNNING

    def test_root_appears_in_snapshot(self):
        t = ActionTracker()
        t.record_start(action_id="a1", task_name="task1")
        root_ids, _children, nodes = t.snapshot()
        assert "a1" in root_ids
        assert "a1" in nodes

    def test_child_appears_under_parent(self):
        t = ActionTracker()
        t.record_start(action_id="root", task_name="root_task")
        t.record_start(action_id="child", task_name="child_task", parent_id="root")
        root_ids, children, _nodes = t.snapshot()
        assert "child" not in root_ids
        assert "child" in children.get("root", [])

    def test_multiple_children(self):
        t = ActionTracker()
        t.record_start(action_id="root", task_name="root_task")
        t.record_start(action_id="c1", task_name="t1", parent_id="root")
        t.record_start(action_id="c2", task_name="t2", parent_id="root")
        t.record_start(action_id="c3", task_name="t3", parent_id="root")
        _, children, _ = t.snapshot()
        assert children["root"] == ["c1", "c2", "c3"]

    def test_duplicate_root_not_added_twice(self):
        t = ActionTracker()
        t.record_start(action_id="a1", task_name="task1")
        # Recording again with same ID overwrites the node but shouldn't
        # duplicate the root_id entry
        t.record_start(action_id="a1", task_name="task1_v2")
        root_ids, _, _ = t.snapshot()
        assert root_ids.count("a1") == 1

    def test_short_name_stored(self):
        t = ActionTracker()
        t.record_start(action_id="a1", task_name="mod.task", short_name="task")
        node = t.get_action("a1")
        assert node.short_name == "task"

    def test_cache_fields(self):
        t = ActionTracker()
        t.record_start(
            action_id="a1",
            task_name="t",
            cache_enabled=True,
            cache_hit=True,
        )
        node = t.get_action("a1")
        assert node.cache_enabled is True
        assert node.cache_hit is True

    def test_disable_run_cache_stored(self):
        t = ActionTracker()
        t.record_start(
            action_id="a1",
            task_name="t",
            cache_enabled=True,
            cache_hit=True,
            disable_run_cache=True,
        )
        node = t.get_action("a1")
        assert node.cache_enabled is True
        assert node.cache_hit is True
        assert node.disable_run_cache is True

    def test_context_stored(self):
        t = ActionTracker()
        t.record_start(action_id="a1", task_name="t", context={"env": "prod"})
        node = t.get_action("a1")
        assert node.context == {"env": "prod"}

    def test_inputs_safe_json(self):
        obj = object()
        t = ActionTracker()
        t.record_start(action_id="a1", task_name="t", inputs={"key": obj})
        node = t.get_action("a1")
        assert isinstance(node.inputs["key"], str)

    def test_none_inputs_stored_as_none(self):
        t = ActionTracker()
        t.record_start(action_id="a1", task_name="t", inputs=None)
        node = t.get_action("a1")
        assert node.inputs is None

    def test_empty_dict_inputs_stored_as_none(self):
        """Empty dict is falsy, so record_start stores None."""
        t = ActionTracker()
        t.record_start(action_id="a1", task_name="t", inputs={})
        node = t.get_action("a1")
        assert node.inputs is None


# ---------------------------------------------------------------------------
# ActionTracker — record_complete
# ---------------------------------------------------------------------------


class TestTrackerRecordComplete:
    def test_basic_complete(self):
        t = ActionTracker()
        t.record_start(action_id="a1", task_name="t")
        t.record_complete(action_id="a1", outputs={"o0": "result"})
        node = t.get_action("a1")
        assert node.status == ActionStatus.SUCCEEDED
        assert node.end_time is not None
        assert node.outputs == {"o0": "result"}

    def test_complete_unknown_action_is_noop(self):
        t = ActionTracker()
        # Should not raise
        t.record_complete(action_id="nonexistent", outputs="data")

    def test_complete_with_none_outputs(self):
        t = ActionTracker()
        t.record_start(action_id="a1", task_name="t")
        t.record_complete(action_id="a1", outputs=None)
        node = t.get_action("a1")
        assert node.status == ActionStatus.SUCCEEDED
        assert node.outputs is None

    def test_complete_with_literal_string_repr_fallback(self):
        """When literal_string_repr raises, should fall back to _safe_json."""
        t = ActionTracker()
        t.record_start(action_id="a1", task_name="t")
        # Pass a plain dict — literal_string_repr will raise ValueError for unknown type,
        # and the tracker should fall back to _safe_json
        t.record_complete(action_id="a1", outputs={"key": "value"})
        node = t.get_action("a1")
        # Either way, we should get the dict back
        assert node.outputs == {"key": "value"}

    def test_complete_sets_end_time(self):
        t = ActionTracker()
        t.record_start(action_id="a1", task_name="t")
        before = time.monotonic()
        t.record_complete(action_id="a1")
        after = time.monotonic()
        node = t.get_action("a1")
        assert before <= node.end_time <= after


# ---------------------------------------------------------------------------
# ActionTracker — record_failure
# ---------------------------------------------------------------------------


class TestTrackerRecordFailure:
    def test_basic_failure(self):
        t = ActionTracker()
        t.record_start(action_id="a1", task_name="t")
        t.record_failure(action_id="a1", error="boom")
        node = t.get_action("a1")
        assert node.status == ActionStatus.FAILED
        assert node.error == "boom"
        assert node.end_time is not None

    def test_failure_unknown_action_is_noop(self):
        t = ActionTracker()
        t.record_failure(action_id="nonexistent", error="err")

    def test_failure_preserves_inputs(self):
        t = ActionTracker()
        t.record_start(action_id="a1", task_name="t", inputs={"x": 1})
        t.record_failure(action_id="a1", error="err")
        node = t.get_action("a1")
        assert node.inputs == {"x": 1}
        assert node.error == "err"


# ---------------------------------------------------------------------------
# ActionTracker — version
# ---------------------------------------------------------------------------


class TestTrackerVersion:
    def test_starts_at_zero(self):
        t = ActionTracker()
        assert t.version == 0

    def test_increments_on_start(self):
        t = ActionTracker()
        t.record_start(action_id="a1", task_name="t")
        assert t.version == 1

    def test_increments_on_complete(self):
        t = ActionTracker()
        t.record_start(action_id="a1", task_name="t")
        t.record_complete(action_id="a1")
        assert t.version == 2

    def test_increments_on_failure(self):
        t = ActionTracker()
        t.record_start(action_id="a1", task_name="t")
        t.record_failure(action_id="a1", error="err")
        assert t.version == 2

    def test_no_increment_on_noop_complete(self):
        t = ActionTracker()
        t.record_complete(action_id="nonexistent")
        assert t.version == 0

    def test_no_increment_on_noop_failure(self):
        t = ActionTracker()
        t.record_failure(action_id="nonexistent", error="err")
        assert t.version == 0


# ---------------------------------------------------------------------------
# ActionTracker — get_action
# ---------------------------------------------------------------------------


class TestTrackerGetAction:
    def test_returns_none_for_missing(self):
        t = ActionTracker()
        assert t.get_action("nope") is None

    def test_returns_correct_node(self):
        t = ActionTracker()
        t.record_start(action_id="a1", task_name="t1")
        t.record_start(action_id="a2", task_name="t2")
        node = t.get_action("a2")
        assert node.task_name == "t2"


# ---------------------------------------------------------------------------
# ActionTracker — snapshot isolation
# ---------------------------------------------------------------------------


class TestTrackerSnapshot:
    def test_snapshot_returns_copies(self):
        """Mutations to snapshot should not affect the tracker."""
        t = ActionTracker()
        t.record_start(action_id="a1", task_name="t1")
        root_ids, _children, nodes = t.snapshot()
        root_ids.append("injected")
        nodes["injected"] = "fake"
        # Original tracker should be unaffected
        root_ids2, _, nodes2 = t.snapshot()
        assert "injected" not in root_ids2
        assert "injected" not in nodes2

    def test_snapshot_empty(self):
        t = ActionTracker()
        root_ids, children, nodes = t.snapshot()
        assert root_ids == []
        assert children == {}
        assert nodes == {}

    def test_snapshot_multi_level_tree(self):
        t = ActionTracker()
        t.record_start(action_id="root", task_name="root_task")
        t.record_start(action_id="child", task_name="child_task", parent_id="root")
        t.record_start(action_id="grandchild", task_name="gc_task", parent_id="child")
        root_ids, children, nodes = t.snapshot()
        assert root_ids == ["root"]
        assert children["root"] == ["child"]
        assert children["child"] == ["grandchild"]
        assert len(nodes) == 3


# ---------------------------------------------------------------------------
# ActionTracker — thread safety
# ---------------------------------------------------------------------------


class TestTrackerThreadSafety:
    def test_concurrent_writes(self):
        """Many threads recording simultaneously should not corrupt state."""
        t = ActionTracker()
        n_threads = 20
        n_per_thread = 50
        barrier = threading.Barrier(n_threads)

        def worker(tid: int):
            barrier.wait()
            for i in range(n_per_thread):
                aid = f"t{tid}-{i}"
                t.record_start(action_id=aid, task_name=f"task-{tid}-{i}")
                t.record_complete(action_id=aid, outputs={"i": i})

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(n_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        _, _, nodes = t.snapshot()
        assert len(nodes) == n_threads * n_per_thread
        assert t.version == n_threads * n_per_thread * 2  # start + complete each


# ---------------------------------------------------------------------------
# TaskCallSequencer
# ---------------------------------------------------------------------------


class TestTaskCallSequencer:
    def test_basic_sequencing(self):
        from flyte._internal.controllers import TaskCallSequencer

        s = TaskCallSequencer()
        assert s.next_seq(type("T", (), {"name": "my_task"})(), "parent") == 1
        assert s.next_seq(type("T", (), {"name": "my_task"})(), "parent") == 2
        assert s.next_seq(type("T", (), {"name": "my_task"})(), "parent") == 3

    def test_different_tasks_independent(self):
        from flyte._internal.controllers import TaskCallSequencer

        s = TaskCallSequencer()
        assert s.next_seq(type("A", (), {"name": "task_a"})(), "p") == 1
        assert s.next_seq(type("B", (), {"name": "task_b"})(), "p") == 1
        assert s.next_seq(type("A", (), {"name": "task_a"})(), "p") == 2

    def test_different_parents_independent(self):
        from flyte._internal.controllers import TaskCallSequencer

        s = TaskCallSequencer()
        task = type("T", (), {"name": "t"})()
        assert s.next_seq(task, "parent1") == 1
        assert s.next_seq(task, "parent2") == 1

    def test_clear(self):
        from flyte._internal.controllers import TaskCallSequencer

        s = TaskCallSequencer()
        task = type("T", (), {"name": "t"})()
        s.next_seq(task, "p1")
        s.next_seq(task, "p1")
        s.clear("p1")
        assert s.next_seq(task, "p1") == 1

    def test_function_uses_dunder_name(self):
        from flyte._internal.controllers import TaskCallSequencer

        s = TaskCallSequencer()

        def my_func():
            pass

        assert s.next_seq(my_func, "p") == 1
        assert s.next_seq(my_func, "p") == 2

    def test_object_without_name_uses_id(self):
        from flyte._internal.controllers import TaskCallSequencer

        s = TaskCallSequencer()
        obj = object()
        assert s.next_seq(obj, "p") == 1
        assert s.next_seq(obj, "p") == 2


# ---------------------------------------------------------------------------
# ActionTracker — group nesting
# ---------------------------------------------------------------------------


class TestTrackerGroupNesting:
    def test_group_creates_virtual_node(self):
        t = ActionTracker()
        t.record_start(action_id="root", task_name="root_task")
        t.record_start(action_id="c1", task_name="child", parent_id="root", group="my_group")
        _, _children, nodes = t.snapshot()
        group_key = "__group__root__my_group"
        assert group_key in nodes
        assert nodes[group_key].task_name == "my_group"
        assert nodes[group_key].parent_id == "root"

    def test_group_node_under_parent(self):
        t = ActionTracker()
        t.record_start(action_id="root", task_name="root_task")
        t.record_start(action_id="c1", task_name="child", parent_id="root", group="grp")
        _, children, _ = t.snapshot()
        group_key = "__group__root__grp"
        assert group_key in children["root"]
        assert "c1" not in children["root"]

    def test_child_nested_under_group(self):
        t = ActionTracker()
        t.record_start(action_id="root", task_name="root_task")
        t.record_start(action_id="c1", task_name="child", parent_id="root", group="grp")
        _, children, nodes = t.snapshot()
        group_key = "__group__root__grp"
        assert "c1" in children[group_key]
        assert nodes["c1"].parent_id == group_key

    def test_multiple_children_share_group_node(self):
        t = ActionTracker()
        t.record_start(action_id="root", task_name="root_task")
        t.record_start(action_id="c1", task_name="child1", parent_id="root", group="grp")
        t.record_start(action_id="c2", task_name="child2", parent_id="root", group="grp")
        t.record_start(action_id="c3", task_name="child3", parent_id="root", group="grp")
        _, children, _nodes = t.snapshot()
        group_key = "__group__root__grp"
        # Only one group node created
        assert children["root"].count(group_key) == 1
        assert children[group_key] == ["c1", "c2", "c3"]

    def test_different_groups_create_separate_nodes(self):
        t = ActionTracker()
        t.record_start(action_id="root", task_name="root_task")
        t.record_start(action_id="c1", task_name="child1", parent_id="root", group="grp_a")
        t.record_start(action_id="c2", task_name="child2", parent_id="root", group="grp_b")
        _, children, nodes = t.snapshot()
        assert "__group__root__grp_a" in nodes
        assert "__group__root__grp_b" in nodes
        assert children["__group__root__grp_a"] == ["c1"]
        assert children["__group__root__grp_b"] == ["c2"]

    def test_no_group_nests_directly_under_parent(self):
        """Without group, children nest directly under parent (existing behavior)."""
        t = ActionTracker()
        t.record_start(action_id="root", task_name="root_task")
        t.record_start(action_id="c1", task_name="child", parent_id="root")
        _, children, _ = t.snapshot()
        assert "c1" in children["root"]

    def test_group_without_parent_is_root(self):
        """group is ignored when parent_id is None (root action)."""
        t = ActionTracker()
        t.record_start(action_id="root", task_name="root_task", group="grp")
        root_ids, _, nodes = t.snapshot()
        assert "root" in root_ids
        # No virtual group node created
        assert "__group__None__grp" not in nodes

    def test_group_status_all_succeeded(self):
        t = ActionTracker()
        t.record_start(action_id="root", task_name="root_task")
        t.record_start(action_id="c1", task_name="t1", parent_id="root", group="grp")
        t.record_start(action_id="c2", task_name="t2", parent_id="root", group="grp")
        t.record_complete(action_id="c1")
        t.record_complete(action_id="c2")
        group_node = t.get_action("__group__root__grp")
        assert group_node.status == ActionStatus.SUCCEEDED
        assert group_node.end_time is not None

    def test_group_status_running_while_child_running(self):
        t = ActionTracker()
        t.record_start(action_id="root", task_name="root_task")
        t.record_start(action_id="c1", task_name="t1", parent_id="root", group="grp")
        t.record_start(action_id="c2", task_name="t2", parent_id="root", group="grp")
        t.record_complete(action_id="c1")
        # c2 still running
        group_node = t.get_action("__group__root__grp")
        assert group_node.status == ActionStatus.RUNNING

    def test_group_status_failed_when_child_fails(self):
        t = ActionTracker()
        t.record_start(action_id="root", task_name="root_task")
        t.record_start(action_id="c1", task_name="t1", parent_id="root", group="grp")
        t.record_start(action_id="c2", task_name="t2", parent_id="root", group="grp")
        t.record_failure(action_id="c1", error="boom")
        t.record_complete(action_id="c2")
        group_node = t.get_action("__group__root__grp")
        assert group_node.status == ActionStatus.FAILED

    def test_group_end_time_is_max_child_end_time(self):
        t = ActionTracker()
        t.record_start(action_id="root", task_name="root_task")
        t.record_start(action_id="c1", task_name="t1", parent_id="root", group="grp")
        t.record_complete(action_id="c1")
        t.record_start(action_id="c2", task_name="t2", parent_id="root", group="grp")
        t.record_complete(action_id="c2")
        group_node = t.get_action("__group__root__grp")
        c2_node = t.get_action("c2")
        assert group_node.end_time == c2_node.end_time

    def test_group_field_stored_on_node(self):
        t = ActionTracker()
        t.record_start(action_id="root", task_name="root_task")
        t.record_start(action_id="c1", task_name="child", parent_id="root", group="my_grp")
        node = t.get_action("c1")
        assert node.group == "my_grp"

    def test_mixed_grouped_and_ungrouped(self):
        """Grouped and ungrouped children coexist under the same parent."""
        t = ActionTracker()
        t.record_start(action_id="root", task_name="root_task")
        t.record_start(action_id="c1", task_name="t1", parent_id="root", group="grp")
        t.record_start(action_id="c2", task_name="t2", parent_id="root")  # no group
        _, children, _ = t.snapshot()
        group_key = "__group__root__grp"
        assert group_key in children["root"]
        assert "c2" in children["root"]
        assert "c1" in children[group_key]
