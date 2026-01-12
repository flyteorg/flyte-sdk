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


@pytest.mark.asyncio
async def test_trace_uses_own_action_id():
    """Test that traced functions execute with their own action ID, not the parent's"""
    await flyte.init.aio()

    # Track action IDs during execution
    parent_action_name = None
    trace_action_name = None

    @env.task
    async def parent_task() -> str:
        nonlocal parent_action_name, trace_action_name
        # Capture parent's action ID
        parent_action_name = flyte.ctx().action.name

        # Call traced function
        await traced_func()

        # Return both for verification
        return f"{parent_action_name}:{trace_action_name}"

    @flyte.trace
    async def traced_func() -> int:
        nonlocal trace_action_name
        # Capture trace's action ID during execution
        trace_action_name = flyte.ctx().action.name
        return 42

    run = flyte.run(parent_task)
    run.wait()

    # Verify that the trace had a different action ID than the parent
    assert parent_action_name is not None, "Parent action name should be captured"
    assert trace_action_name is not None, "Trace action name should be captured"
    assert (
        trace_action_name != parent_action_name
    ), f"Trace should have different action ID than parent. Trace: {trace_action_name}, Parent: {parent_action_name}"
