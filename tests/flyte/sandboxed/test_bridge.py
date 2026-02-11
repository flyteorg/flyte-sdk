"""Tests for the ExternalFunctionBridge."""

from flyte.sandboxed._bridge import ExternalFunctionBridge


class TestExternalFunctionBridge:
    def test_init_merges_refs(self):
        task_refs = {"add": "task_obj"}
        trace_refs = {"traced": "trace_obj"}
        durable_refs = {"durable_time": "durable_obj"}

        bridge = ExternalFunctionBridge(
            task_refs=task_refs,
            trace_refs=trace_refs,
            durable_refs=durable_refs,
        )

        assert bridge._all_refs == {
            "add": "task_obj",
            "traced": "trace_obj",
            "durable_time": "durable_obj",
        }

    def test_init_empty_refs(self):
        bridge = ExternalFunctionBridge(
            task_refs={},
            trace_refs={},
            durable_refs={},
        )
        assert bridge._all_refs == {}
