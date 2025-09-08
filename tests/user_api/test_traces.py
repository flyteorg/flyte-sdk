import pytest

import flyte

env = flyte.TaskEnvironment(
    name="traces",
)


@flyte.trace
async def square(x: int) -> int:
    return x**2


@flyte.trace
async def print_now(x: int, y: int):
    print(f"x: {x}, y: {y}", flush=True)


@flyte.trace
async def no_inputs() -> int:
    return 42


@flyte.trace
async def no_inputs_outputs():
    print("Hello World", flush=True)


@env.task
async def traces(n: int = 3) -> int:
    total = 0
    for i in range(n):
        sq = await square(i)
        await print_now(i, sq)
        total += sq

    no_input_val = await no_inputs()
    print(f"No input val: {no_input_val}", flush=True)
    await no_inputs_outputs()
    return total


@env.task
async def traces_loop(n: int = 3) -> int:
    total = 0
    for i in range(n):
        sq = await square(i)
        total += sq
    return total


@pytest.mark.asyncio
async def test_traces_loop():
    await flyte.init.aio()
    run = flyte.run(traces_loop, n=3)
    print(run.name)
    print(run.url)
    run.wait()
    assert run.outputs() == 5


@pytest.mark.asyncio
async def test_traces():
    await flyte.init.aio()
    run = flyte.run(traces, n=5)
    print(run.name)
    print(run.url)
    run.wait()
    assert run.outputs() == 30
