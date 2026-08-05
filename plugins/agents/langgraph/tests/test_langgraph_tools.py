"""Unit tests for the LangGraph adapter's tool bridge (no network / no controller)."""

from unittest.mock import AsyncMock, patch

import flyte
import pytest
from flyteplugins.agents.core import ToolTaskResolver
from langchain_core.tools import BaseTool

from flyteplugins.agents.langgraph import tool


def test_task_becomes_langchain_tool_with_resolver():
    env = flyte.TaskEnvironment("lg_tools_a")

    @tool
    @env.task
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"sunny in {city}"

    # The tool is a first-class LangChain tool carrying the task + resolver.
    assert isinstance(get_weather, BaseTool)
    assert get_weather.name == "get_weather"
    assert get_weather.__name__ == "get_weather"
    assert get_weather.__wrapped_task__ is get_weather.task
    assert isinstance(get_weather.task.task_resolver, ToolTaskResolver)
    # The args schema is inferred from the task's typed signature.
    assert "city" in get_weather.args_schema.model_fields


def test_bind_tools_and_tool_node_accept_the_tool():
    from langgraph.prebuilt import ToolNode

    env = flyte.TaskEnvironment("lg_tools_b")

    @tool
    @env.task
    def get_weather(city: str) -> str:
        """Get weather."""
        return city

    # A real LangGraph ToolNode accepts the tool.
    node = ToolNode([get_weather])
    assert node is not None


@pytest.mark.asyncio
async def test_tool_dispatches_to_task_aio():
    env = flyte.TaskEnvironment("lg_tools_c")

    @tool
    @env.task
    def multiply(a: int, b: int) -> int:
        """Multiply."""
        return a * b

    with patch.object(multiply.task, "aio", new_callable=AsyncMock, return_value=42) as mock_aio:
        # Invoking the tool the way LangGraph does dispatches to task.aio().
        result = await multiply.ainvoke({"a": 6, "b": 7})

    mock_aio.assert_awaited_once_with(a=6, b=7)
    assert result == "42"


def test_bare_and_parametrized_decorator_forms():
    env = flyte.TaskEnvironment("lg_tools_d")

    @tool
    @env.task
    def a(x: int) -> int:
        """A."""
        return x

    assert a.name == "a"
    assert a.__name__ == "a"

    @tool(name="bee")
    @env.task
    def b(x: int) -> int:
        """B."""
        return x

    assert b.name == "bee"
    assert b.__name__ == "bee"
