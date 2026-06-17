"""Unit tests for the OpenAI adapter's tool bridge (no network / no controller)."""

from unittest.mock import AsyncMock, patch

import flyte
import pytest
from agents import FunctionTool as OpenAIFunctionTool

from flyteplugins.agents.openai import FunctionTool, function_tool


def test_task_becomes_flyte_backed_tool():
    env = flyte.TaskEnvironment("tools_a")

    @env.task
    def my_task(prompt: str) -> str:
        """Echo the prompt."""
        return prompt

    tool = function_tool(my_task)
    assert isinstance(tool, FunctionTool)
    assert tool.task is my_task
    assert tool.report == my_task.report
    assert tool.native_interface is my_task.native_interface
    assert tool.name == "my_task"
    assert "prompt" in tool.params_json_schema.get("properties", {})


def test_plain_function_uses_native_tool():
    def f(x: int) -> int:
        """Double."""
        return x * 2

    tool = function_tool(f)
    assert isinstance(tool, OpenAIFunctionTool)
    assert not isinstance(tool, FunctionTool)


def test_trace_helper_uses_native_tool():
    @flyte.trace
    def f(x: int) -> int:
        """Double."""
        return x * 2

    tool = function_tool(f)
    assert isinstance(tool, OpenAIFunctionTool)
    assert not isinstance(tool, FunctionTool)


def test_bare_and_parametrized_decorator_forms():
    env = flyte.TaskEnvironment("tools_b")

    @function_tool
    @env.task
    def a(x: int) -> int:
        """A."""
        return x

    assert isinstance(a, FunctionTool)

    @function_tool(name_override="bee")
    @env.task
    def b(x: int) -> int:
        """B."""
        return x

    assert isinstance(b, FunctionTool)
    assert b.name == "bee"


@pytest.mark.asyncio
async def test_execute_dispatches_to_task_aio():
    env = flyte.TaskEnvironment("tools_c")

    @env.task
    def multiply(a: int, b: int) -> int:
        """Multiply."""
        return a * b

    tool = function_tool(multiply)
    with patch.object(tool.task, "aio", new_callable=AsyncMock, return_value=42) as mock_aio:
        result = await tool.execute(a=6, b=7)

    mock_aio.assert_awaited_once_with(a=6, b=7)
    assert result == 42


def test_function_tool_attaches_resolver_so_task_does_not_self_recurse():
    """Regression: ``@function_tool`` on ``@env.task`` shadows the task at module
    scope. Without a ``__wrapped_task__`` hook + ``ToolTaskResolver``, the worker
    loads the tool, calls ``FunctionTool.execute``, and the task re-dispatches
    itself indefinitely. Assert both guards are wired up."""
    from flyteplugins.agents.openai._tools import ToolTaskResolver

    env = flyte.TaskEnvironment("tools_resolver")

    @function_tool
    @env.task
    def my_task(city: str) -> str:
        """Echo."""
        return city

    assert isinstance(my_task, FunctionTool)
    # The tool exposes the real task for the resolver to recover on the worker.
    assert my_task.__wrapped_task__ is my_task.task
    # …and the task is pointed at the recovering resolver.
    assert isinstance(my_task.task.task_resolver, ToolTaskResolver)
