import pathlib
from typing import Any, Dict, List

import aiofiles
import pytest

import flyte
from flyte._internal import create_controller
from flyte._internal.runtime import io
from flyte._internal.runtime.convert import convert_from_native_to_inputs, convert_outputs_to_native
from flyte._internal.runtime.entrypoints import load_and_run_task
from flyte.models import ActionID, RawDataPath

env = flyte.TaskEnvironment(name="test")


@env.task
def sync_task1(v: str) -> str:
    return f"Hello, world {v}!"


@env.task
def sync_parent_task(i: int) -> List[str]:
    vals = []
    for i in range(i):
        vals.append(sync_task1(str(i)))
    return vals


def test_parent_action_raw():
    result = sync_parent_task(3)
    assert result == ["Hello, world 0!", "Hello, world 1!", "Hello, world 2!"]


def test_typing():
    assert sync_parent_task._call_as_synchronous is True


def test_parent_action_local():
    flyte.init()
    result = flyte.run(sync_parent_task, 3)
    assert result.outputs()[0] == ["Hello, world 0!", "Hello, world 1!", "Hello, world 2!"]


@env.task
def sync_runtime_probe(n: int) -> Dict[str, Any]:
    """A sync body that checks what the runtime gives it: task context, a plain (loop-less) thread,
    the ability to run its own asyncio loop, and a nested sync child call."""
    import asyncio
    import threading

    try:
        asyncio.get_running_loop()
        has_running_loop = True
    except RuntimeError:
        has_running_loop = False

    async def _inner() -> int:
        return n * 2

    ctx = flyte.ctx()
    return {
        "action": ctx.action.name if ctx else None,
        "thread": threading.current_thread().name,
        "has_running_loop": has_running_loop,
        "asyncio_run": asyncio.run(_inner()),
        "child": sync_task1("nested"),
    }


@pytest.mark.asyncio
async def test_sync_task_through_runtime_taskrunner(tmp_path):
    """Drive a sync task through the real remote-runtime entrypoint (load_and_run_task -> taskrunner ->
    execute -> run_sync_in_thread), not the local controller's own path."""
    await flyte.init.aio()
    inputs = await convert_from_native_to_inputs(sync_runtime_probe.native_interface, 21)
    input_path = tmp_path / "inputs.pb"
    async with aiofiles.open(input_path, "wb") as f:
        await f.write(inputs.proto_inputs.SerializeToString())

    await load_and_run_task(
        resolver="flyte._internal.resolvers.default.DefaultTaskResolver",
        resolver_args=["mod", "tests.flyte.test_sync_tasks", "instance", "sync_runtime_probe"],
        action=ActionID(name="probe_action", run_name="probe_run"),
        raw_data_path=RawDataPath(path="raw_data_path"),
        input_path=str(input_path),
        output_path=str(tmp_path),
        run_base_dir=str(tmp_path),
        version="v1",
        controller=create_controller("local"),
    )
    outputs_path = pathlib.Path(io.outputs_path(str(tmp_path)))
    assert outputs_path.is_file(), "task did not produce outputs"
    result = await convert_outputs_to_native(
        sync_runtime_probe.native_interface, outputs=await io.load_outputs(path=str(outputs_path))
    )

    assert result["action"] == "probe_action"  # task context propagated into the body thread
    assert "sync-executor" in result["thread"]  # body ran on the dedicated thread, not the runtime loop
    assert result["has_running_loop"] is False  # plain sync code, no hidden helper loop
    assert result["asyncio_run"] == 42  # asyncio.run() works inside a sync task
    assert result["child"] == "Hello, world nested!"  # nested sync child submitted and awaited
