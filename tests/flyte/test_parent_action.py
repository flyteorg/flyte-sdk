import asyncio
import pathlib
from typing import List

import aiofiles
import pytest

import flyte
from flyte._internal import create_controller
from flyte._internal.runtime import entrypoints, io
from flyte._internal.runtime.convert import convert_from_native_to_inputs, convert_outputs_to_native
from flyte._internal.runtime.entrypoints import load_and_run_task
from flyte._utils import run_coros
from flyte.models import ActionID, RawDataPath

env = flyte.TaskEnvironment(name="test")


@env.task
async def task1(v: str) -> str:
    return f"Hello, world {v}!"


@env.task
async def parent_task(i: int) -> List[str]:
    vals = []
    for i in range(i):
        vals.append(task1(str(i)))
    return await asyncio.gather(*vals)


@pytest.mark.asyncio
async def test_parent_action_raw():
    result = await parent_task(3)
    assert result == ["Hello, world 0!", "Hello, world 1!", "Hello, world 2!"]


@pytest.mark.asyncio
async def test_parent_action_local():
    _ = await parent_task(3)
    await flyte.init.aio()
    result = await flyte.run.aio(parent_task, 3)
    assert result.outputs() == ["Hello, world 0!", "Hello, world 1!", "Hello, world 2!"]


@pytest.mark.asyncio
async def test_parent_action_controller_mock(tmp_path):
    await flyte.init.aio()
    inputs = await convert_from_native_to_inputs(parent_task.native_interface, 3)
    input_path = tmp_path / "inputs.pb"
    async with aiofiles.open(input_path, "wb") as f:
        await f.write(inputs.proto_inputs.SerializeToString())

    await load_and_run_task(
        resolver="flyte._internal.resolvers.default.DefaultTaskResolver",
        resolver_args=["mod", "tests.flyte.test_parent_action", "instance", "parent_task"],
        action=ActionID(name="test_run", run_name="test_root"),
        raw_data_path=RawDataPath(path="raw_data_path"),
        input_path=str(input_path),
        output_path=str(tmp_path),
        run_base_dir=str(tmp_path),
        version="v1",
        controller=create_controller("local"),
    )
    outputs_path = pathlib.Path(io.outputs_path(str(tmp_path)))
    assert outputs_path.exists(), "Outputs path does not exist, not created by the task run."
    assert outputs_path.is_file()
    result = await io.load_outputs(path=str(outputs_path))
    native_result = await convert_outputs_to_native(parent_task.native_interface, outputs=result)
    assert native_result == ["Hello, world 0!", "Hello, world 1!", "Hello, world 2!"]


@pytest.mark.asyncio
async def test_parent_action_failure_controller_mock(tmp_path, monkeypatch):
    def fake_download_and_load_task(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(entrypoints, "_download_and_load_task", fake_download_and_load_task)
    await flyte.init.aio()
    inputs = await convert_from_native_to_inputs(parent_task.native_interface, 3)
    input_path = tmp_path / "inputs.pb"
    async with aiofiles.open(input_path, "wb") as f:
        await f.write(inputs.proto_inputs.SerializeToString())

    async def _main_task_coroutine():
        await asyncio.sleep(0.05)
        await load_and_run_task(
            resolver="flyte._internal.resolvers.default.DefaultTaskResolver",
            resolver_args=["mod", "tests.flyte.test_parent_action", "instance", "parent_task"],
            action=ActionID(name="test_run", run_name="test_root"),
            raw_data_path=RawDataPath(path="raw_data_path"),
            input_path=str(input_path),
            output_path=str(tmp_path),
            run_base_dir=str(tmp_path),
            version="v1",
            controller=create_controller("local"),
        )

    task_coroutine = _main_task_coroutine()

    async def _other_async_fn():
        await asyncio.sleep(10)

    with pytest.raises(RuntimeError):
        await run_coros(task_coroutine, _other_async_fn())
