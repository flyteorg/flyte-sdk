"""Tests for the ExternalFunctionBridge."""

import asyncio
import threading

import pytest

import flyte.sandbox
from flyte.sandbox._bridge import ExternalFunctionBridge, _call_external
from flyte.syncify import syncify


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


class TestCallExternal:
    """`_call_external` dispatches sync work off the loop thread and awaits async work inline."""

    @pytest.mark.asyncio
    async def test_sync_fn_runs_off_loop_thread(self):
        seen: dict[str, threading.Thread] = {}

        def sync_fn(x: int) -> int:
            seen["thread"] = threading.current_thread()
            return x + 1

        assert await _call_external(sync_fn, 1) == 2
        assert seen["thread"] is not threading.current_thread()

    @pytest.mark.asyncio
    async def test_async_fn_runs_on_loop(self):
        seen: dict[str, threading.Thread] = {}

        async def async_fn(x: int) -> int:
            seen["thread"] = threading.current_thread()
            return x + 1

        assert await _call_external(async_fn, 1) == 2
        assert seen["thread"] is threading.current_thread()

    @pytest.mark.asyncio
    async def test_returned_coroutine_is_awaited(self):
        # TaskTemplate.aio() in local mode may hand back an unawaited coroutine
        # from forward(); _call_external must drain it to a concrete value.
        async def inner() -> str:
            return "done"

        def sync_returning_coro():
            return inner()

        assert await _call_external(sync_returning_coro) == "done"


# --- Regression: agent code mode with sync @flyte.trace tools -----------------
#
# With `code_mode=True` the whole sandbox loop can end up running on the
# `flyte_syncify` background loop (any syncified entry point). Sync
# `@flyte.trace` wrappers make *blocking* syncify calls
# (`_fetch_action_outputs` / `_record_trace_action`); if the bridge invokes
# them inline on that same thread, syncify's deadlock detection aborts every
# tool call. The bridge must run sync externals in a worker thread instead.


@syncify
async def _syncified_helper() -> str:
    return "ok"


def blocking_tool(x: int) -> int:
    """A sync tool that blocks on syncify, exactly like a sync @flyte.trace wrapper."""
    assert _syncified_helper() == "ok"
    return x + 1


@syncify
async def _orchestrate_on_syncify_loop(code: str, tools: list, inputs: dict):
    return await flyte.sandbox.orchestrate_local(code, inputs=inputs, tasks=tools)


class TestSyncToolOnSyncifyLoop:
    def test_blocking_sync_tool_does_not_deadlock(self):
        # Before the fix this raised: "Deadlock detected: blocking call used in
        # syncify thread flyte_syncify ... use .aio() if in an async call."
        result = _orchestrate_on_syncify_loop("blocking_tool(x)", [blocking_tool], {"x": 1})
        assert result == 2

    def test_flyte_map_over_blocking_sync_tool_does_not_deadlock(self):
        result = _orchestrate_on_syncify_loop(
            "flyte_map('blocking_tool', xs)",
            [blocking_tool],
            {"xs": [1, 2, 3]},
        )
        assert result == [2, 3, 4]

    def test_event_loop_stays_responsive_while_sync_tool_blocks(self):
        # While a sync tool sleeps in its worker thread, the loop driving the
        # bridge must still be able to run other coroutines.
        import time

        async def heartbeat() -> int:
            ticks = 0
            for _ in range(5):
                await asyncio.sleep(0.01)
                ticks += 1
            return ticks

        def parked(x: int) -> int:
            time.sleep(0.05)
            return x

        async def drive() -> tuple[int, int]:
            return await asyncio.gather(_call_external(parked, 7), heartbeat())

        tool_result, ticks = asyncio.run(drive())
        assert tool_result == 7
        assert ticks == 5
