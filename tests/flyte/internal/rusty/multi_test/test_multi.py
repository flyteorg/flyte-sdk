import asyncio
import pathlib

import pytest

from flyte._internal import Controller
from flyte._internal.runtime import convert, io, rusty
from flyte._task import TaskTemplate


async def run_test(
    tk: TaskTemplate, controller: Controller, i: int, base_dir: pathlib.Path, input_path: str, output_val_expected: int
):
    data_path = base_dir / f"{i}"
    data_path.mkdir(parents=True, exist_ok=True)
    raw_data_path = str(data_path / "raw")
    output_path = str(data_path)
    run_base_dir = str(base_dir)
    version = "v1"
    await rusty.run_task(
        task=tk,
        controller=controller,
        org="flyteorg",
        project="flyteproject",
        domain="development",
        run_name=f"run-{i}",
        name=f"test-task-{i}",
        run_base_dir=run_base_dir,
        raw_data_path=raw_data_path,
        output_path=output_path,
        version=version,
        input_path=input_path,
    )

    outputs = await io.load_outputs(io.outputs_path(output_path))
    v = await convert.convert_outputs_to_native(tk.interface, outputs)
    assert v == f"test-task-{i} - {output_val_expected}"


async def main(n: int = 1000, base_dir: pathlib.Path = pathlib.Path("")) -> None:
    controller = await rusty.create_controller(endpoint="dns:///localhost:8090", insecure=True)
    tk = rusty.load_task(
        "flyte._internal.resolvers.default.DefaultTaskResolver", "mod", "app.tasks", "instance", "square"
    )

    inputs = await convert.convert_from_native_to_inputs(tk.interface, 10)
    input_path = io.inputs_path(base_path=str(base_dir))
    await io.upload_inputs(inputs, input_path)

    i = await io.load_inputs(input_path)
    assert i.proto_inputs.literals[0].value.scalar.primitive.integer == 10, "Input value should be 10"

    task_coros = []
    for i in range(n):
        task_coros.append(run_test(tk, controller, i, base_dir, input_path, 100))

    await asyncio.gather(*task_coros)


@pytest.mark.asyncio
async def test_multi(tmp_path: pathlib.Path):
    base_dir = pathlib.Path(tmp_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    await main(n=1000, base_dir=base_dir)
    print("Multi test completed successfully.")
