"""Tests for the ExternalFunctionBridge."""

import asyncio
import threading
import time

import pytest

import flyte.sandbox
from flyte.io import Dir, File
from flyte.sandbox._bridge import ExternalFunctionBridge, _call_external
from flyte.sandbox._config import SandboxedConfig
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


class TestFilesAndDirs:
    """A File or Dir is a named tuple of its fields in the sandbox, as the remote worker builds it."""

    @pytest.mark.asyncio
    async def test_source_reads_fields_and_passes_them_back(self):
        f = File(path="s3://bucket/data/report.csv", format="csv", hash="abc")
        d = Dir(path="s3://bucket/data/shards/")
        seen = []

        def make():
            return [f, d]

        def take(file, directory):
            seen.extend([file, directory])
            return file

        bridge = ExternalFunctionBridge(task_refs={}, trace_refs={"make": make, "take": take}, durable_refs={})
        code = "f, d = make()\nkept = take(f, d)\n[kept, f.path, f.name, f.format, f.hash, d.name, d.hash]"
        result = await bridge.execute_monty(SandboxedConfig(), code, {})

        assert result == [f, "s3://bucket/data/report.csv", "report.csv", "csv", "abc", "shards", None]
        assert seen == [f, d]
        assert type(seen[0]) is File and type(seen[1]) is Dir

    @pytest.mark.asyncio
    async def test_inputs_are_named_tuples(self):
        bridge = ExternalFunctionBridge(task_refs={}, trace_refs={}, durable_refs={})
        result = await bridge.execute_monty(SandboxedConfig(), "f.path", {"f": File(path="s3://bucket/key")})
        assert result == "s3://bucket/key"


class TestOpaqueValues:
    """Dataframes, Pydantic models and dataclasses are opaque in the sandbox: passed on, never opened."""

    @pytest.mark.asyncio
    async def test_passed_on_as_the_same_objects(self):
        import dataclasses

        import pydantic

        from flyte.io import DataFrame

        class Volume(pydantic.BaseModel):
            name: str

        @dataclasses.dataclass
        class Config:
            lr: float

        df = DataFrame.from_existing_remote("s3://bucket/df.parquet", format="parquet")
        volume, config = Volume(name="v"), Config(lr=0.1)
        seen = []

        def make():
            return {"df": df, "volume": volume, "configs": [config]}

        def take(*values):
            seen.extend(values)
            return len(values)

        bridge = ExternalFunctionBridge(task_refs={}, trace_refs={"make": make, "take": take}, durable_refs={})
        code = (
            "made = make()\n"
            "v = made['volume']\n"
            "names = [repr(v).startswith('Opaque(literal='), len(v)]\n"
            "take(made['df'], v, made['configs'][0])\n"
            "[v, names]"
        )
        result = await bridge.execute_monty(SandboxedConfig(), code, {})

        assert seen[0] is df and seen[1] is volume and seen[2] is config
        assert result[0] is volume
        assert result[1] == [True, 1]

    @pytest.mark.asyncio
    async def test_inputs_cannot_be_opened(self):
        import pydantic

        class Volume(pydantic.BaseModel):
            name: str

        bridge = ExternalFunctionBridge(task_refs={}, trace_refs={}, durable_refs={})
        with pytest.raises(Exception, match="name"):
            await bridge.execute_monty(SandboxedConfig(), "v.name", {"v": Volume(name="v")})


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


def _add(a: int, b: int) -> int:
    return a + b


def _run_with_tool(code: str, **kwargs):
    """Run *code* with a tool attached, which routes it through the bridge."""
    return asyncio.run(flyte.sandbox.orchestrate_local(code, inputs={}, tasks=[_add], **kwargs))


class TestOsCallsThroughBridge:
    """OS calls reach the bridge as function snapshots, alongside tool calls."""

    pydantic_monty = pytest.importorskip("pydantic_monty")

    def test_time_sleep_waits_then_continues(self):
        code = "import time\ntime.sleep(0.2)\n_add(1, 2)"
        start = time.monotonic()
        assert _run_with_tool(code) == 3
        assert time.monotonic() - start >= 0.2

    def test_asyncio_sleep_waits_then_continues(self):
        code = "import asyncio\nawait asyncio.sleep(0.2)\n_add(1, 2)"
        start = time.monotonic()
        assert _run_with_tool(code) == 3
        assert time.monotonic() - start >= 0.2

    def test_sleep_cannot_outlast_the_timeout(self):
        code = "import time\ntime.sleep(5)\n_add(1, 2)"
        start = time.monotonic()
        with pytest.raises(self.pydantic_monty.MontyRuntimeError, match="TimeoutError"):
            _run_with_tool(code, timeout_ms=200)
        assert time.monotonic() - start < 3

    def test_repeated_sleeps_share_one_budget(self):
        code = "import time\ntime.sleep(0.15)\ntime.sleep(0.15)\n_add(1, 2)"
        with pytest.raises(self.pydantic_monty.MontyRuntimeError, match="TimeoutError"):
            _run_with_tool(code, timeout_ms=200)

    def test_environment_access_fails_inside_the_sandbox(self):
        # Declined rather than mistaken for an unknown tool, so the sandbox
        # code can handle it like any other exception.
        code = "import os\ntry:\n    os.environ\n    r = 'leaked'\nexcept RuntimeError:\n    r = _add(1, 2)\nr"
        assert _run_with_tool(code) == 3

    def test_file_access_is_denied(self):
        with pytest.raises(self.pydantic_monty.MontyRuntimeError, match="PermissionError"):
            _run_with_tool("open('/etc/hosts').read()")


class TestAwaitableTaskCalls:
    """With `awaitable_tasks`, a task call returns a future the source awaits, so calls overlap."""

    @staticmethod
    def _run(code: str, task_refs, **inputs):
        bridge = ExternalFunctionBridge(task_refs=task_refs, trace_refs={}, durable_refs={}, awaitable_tasks=True)
        return asyncio.run(bridge.execute_monty(None, code, inputs))

    def test_gathered_calls_run_at_once(self):
        state = {"running": 0, "most": 0}

        async def slow(i: int) -> int:
            state["running"] += 1
            state["most"] = max(state["most"], state["running"])
            await asyncio.sleep(0.05)
            state["running"] -= 1
            return i * 10

        code = """
import asyncio

async def main(n):
    return await asyncio.gather(*[slow(i=i) for i in range(n)])

await main(n)
"""
        assert self._run(code, {"slow": slow}, n=4) == [0, 10, 20, 30]
        assert state["most"] == 4

    def test_a_call_awaited_alone_returns_its_value(self):
        async def double(x: int) -> int:
            return x * 2

        code = """
async def main(x):
    return await double(x=x) + await double(x=1)

await main(x)
"""
        assert self._run(code, {"double": double}, x=5) == 12

    def test_a_raised_in_sandbox_failure_is_caught_where_it_is_awaited(self):
        from flyte.sandbox._bridge import RaiseInSandbox

        async def fails() -> int:
            raise RaiseInSandbox(ValueError("no"))

        code = """
async def main():
    try:
        return await fails()
    except ValueError as error:
        return "caught " + str(error)

await main()
"""
        assert self._run(code, {"fails": fails}) == "caught no"

    def test_other_failures_fail_the_run(self):
        async def fails() -> int:
            raise KeyError("boom")

        code = """
import asyncio

async def main():
    return await asyncio.gather(fails(), fails())

await main()
"""
        with pytest.raises(KeyError, match="boom"):
            self._run(code, {"fails": fails})

    def test_waiting_gives_the_worker_back(self):
        """More sources wait on children that need workers of their own than the pool has workers.

        Each of the parent's children waits on a grandchild that runs in a sandbox too. Were the
        waiting sources to keep their workers, the grandchildren would wait forever for one.
        """
        import os

        one_call = "async def main(i):\n    return await {name}(i=i)\n\nawait main(i)"

        def sandboxed(name, fn):
            async def run(i: int) -> int:
                bridge = ExternalFunctionBridge(
                    task_refs={name: fn}, trace_refs={}, durable_refs={}, awaitable_tasks=True
                )
                return await bridge.execute_monty(None, one_call.format(name=name), {"i": i})

            return run

        async def leaf(i: int) -> int:
            return i

        child = sandboxed("grandchild", sandboxed("leaf", leaf))
        width = (os.cpu_count() or 1) + 2
        code = """
import asyncio

async def main(n):
    return sum(await asyncio.gather(*[child(i=i) for i in range(n)]))

await main(n)
"""

        async def run():
            bridge = ExternalFunctionBridge(
                task_refs={"child": child}, trace_refs={}, durable_refs={}, awaitable_tasks=True
            )
            return await asyncio.wait_for(bridge.execute_monty(None, code, {"n": width}), timeout=60)

        assert asyncio.run(run()) == sum(range(width))


def test_the_sleep_budget_holds_across_the_waits_that_give_the_worker_back():
    """A paused source is reloaded on a fresh worker, which is given only the sleep left."""
    from flyte.sandbox._config import SandboxedConfig

    async def tick() -> int:
        return 1

    code = """
import asyncio

async def main():
    total = 0
    for i in range(6):
        await asyncio.sleep(0.3)
        total = total + await tick()
    return total

await main()
"""
    bridge = ExternalFunctionBridge(task_refs={"tick": tick}, trace_refs={}, durable_refs={}, awaitable_tasks=True)
    with pytest.raises(Exception, match="sleep limit exceeded"):
        asyncio.run(bridge.execute_monty(SandboxedConfig(timeout_ms=1000), code, {}))
