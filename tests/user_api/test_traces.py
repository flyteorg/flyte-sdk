from unittest.mock import AsyncMock, Mock, patch

import pytest

import flyte
import flyte.errors
from flyte._context import Context, ContextData
from flyte._internal.controllers._trace import TraceInfo
from flyte.models import ActionID, NativeInterface

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
    assert run.outputs()[0] == 5


@pytest.mark.asyncio
async def test_traces():
    await flyte.init.aio()
    run = flyte.run(traces, n=5)
    print(run.name)
    print(run.url)
    run.wait()
    assert run.outputs()[0] == 30


@pytest.mark.asyncio
async def test_trace_async_function_task_context_is_none():
    """Test that RuntimeSystemError is raised when task_context is None in async function"""

    @flyte.trace
    async def test_func(x: int) -> int:
        return x * 2

    # Create a mock controller
    mock_controller = AsyncMock()
    trace_info = TraceInfo(
        name="test_func",
        action=ActionID(name="test_trace"),
        interface=NativeInterface(inputs={"x": int}, outputs={"return": int}),
        inputs_path="test",
    )
    mock_controller.get_action_outputs = AsyncMock(return_value=(trace_info, False))

    # Create a context where is_task_context() returns True but task_context is None
    ctx_data = ContextData(
        task_context=Mock()
    )  # Initially set to trigger is_task_context
    test_ctx = Context(data=ctx_data)

    with (
        patch("flyte._trace.internal_ctx", return_value=test_ctx),
        patch(
            "flyte._internal.controllers.get_controller", return_value=mock_controller
        ),
    ):
        # Now set task_context to None to trigger the error at line 47-48
        test_ctx._data = ContextData(task_context=None)

        with pytest.raises(flyte.errors.RuntimeSystemError) as exc_info:
            await test_func(5)

        assert exc_info.value.code == "BadContext"
        assert "Task context not initialized" in str(exc_info.value)


@pytest.mark.asyncio
async def test_trace_async_function_uses_trace_action_id():
    """Test that traced async function executes with trace's action ID in context"""
    from flyte._context import internal_ctx
    from flyte.models import RawDataPath, TaskContext
    from flyte.report import Report

    parent_action_id = ActionID(name="parent_action")
    trace_action_id = ActionID(name="trace_action")

    action_id_during_execution = None

    @flyte.trace
    async def test_func(x: int) -> int:
        nonlocal action_id_during_execution
        # Capture the action ID during execution
        ctx = internal_ctx()
        action_id_during_execution = ctx.data.task_context.action
        return x * 2

    # Create a mock controller
    mock_controller = AsyncMock()
    trace_info = TraceInfo(
        name="test_func",
        action=trace_action_id,  # Trace has its own action ID
        interface=NativeInterface(inputs={"x": int}, outputs={"return": int}),
        inputs_path="test",
    )
    mock_controller.get_action_outputs = AsyncMock(return_value=(trace_info, False))
    mock_controller.record_trace = AsyncMock()

    # Create parent task context with parent action ID
    parent_task_context = TaskContext(
        action=parent_action_id,
        run_base_dir="test",
        output_path="test",
        raw_data_path=RawDataPath(path=""),
        version="",
        report=Report("test"),
    )

    with patch(
        "flyte._internal.controllers.get_controller", return_value=mock_controller
    ):
        ctx = internal_ctx()
        test_ctx = ctx.replace_task_context(parent_task_context)

        async with test_ctx:
            result = await test_func(5)

        # Verify the function executed with trace's action ID
        assert action_id_during_execution == trace_action_id
        assert action_id_during_execution != parent_action_id
        assert result == 10

        # Verify record_trace was called
        mock_controller.record_trace.assert_called_once()
