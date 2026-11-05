import asyncio
import pathlib
import threading

import pytest

import flyte
from flyte._internal.controllers import create_controller
from flyte._internal.runtime import convert, io, rusty

env = flyte.TaskEnvironment(name="rusty-cancel")
started = threading.Event()
release = threading.Event()


@env.task
def wait_for_release(x: int) -> int:
    started.set()
    release.wait(10)
    return x


@pytest.mark.asyncio
async def test_aborted_sync_task_holds_its_slot_until_its_thread_returns(tmp_path: pathlib.Path):
    """The actor executor cancels the run_task task on abort and frees the slot once it finishes."""
    inputs = await convert.convert_from_native_to_inputs(wait_for_release.interface, 1)
    input_path = io.inputs_path(base_path=str(tmp_path))
    await io.upload_inputs(inputs, input_path)

    task = asyncio.create_task(
        rusty.run_task(
            task=wait_for_release,
            controller=create_controller("local"),
            org="flyteorg",
            project="flyteproject",
            domain="development",
            run_name="run-1",
            name="a0",
            raw_data_path=str(tmp_path / "raw"),
            output_path=str(tmp_path / "out"),
            run_base_dir=str(tmp_path),
            version="v1",
            input_path=input_path,
        )
    )
    assert await asyncio.to_thread(started.wait, 10)

    task.cancel()
    done, _ = await asyncio.wait([task], timeout=0.5)
    assert not done, "the aborted task must keep running while its thread runs"

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
