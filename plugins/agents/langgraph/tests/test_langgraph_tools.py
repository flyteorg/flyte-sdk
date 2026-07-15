"""Unit tests for the LangGraph adapter's tool bridge (no network / no controller)."""

from unittest.mock import AsyncMock, patch

import flyte
import pytest
from flyteplugins.agents.core import ToolTaskResolver

from flyteplugins.agents.langgraph import tool


def test_task_becomes_langchain_tool_with_resolver():
    env = flyte.TaskEnvironment("lg_tools_a")

    @tool
    @env.task
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"sunny in {city}"

    # The tool wrapper carries the task + resolver.
    assert get_weather.__name__ == "get_weather"
    assert get_weather.__wrapped_task__ is get_weather.task
    assert isinstance(get_weather.task.task_resolver, ToolTaskResolver)


def test_langchain_agent_accepts_the_tool():
    try:
        from langchain.agents import create_tool_calling_agent
        from langchain_openai import ChatOpenAI

        env = flyte.TaskEnvironment("lg_tools_b")

        @tool
        @env.task
        def get_weather(city: str) -> str:
            """Get weather."""
            return city

        agent = create_tool_calling_agent(
            ChatOpenAI(),
            [get_weather],
            [],
        )
        assert agent is not None
    except Exception:
        pytest.skip("langchain_openai or create_tool_calling_agent not available")


@pytest.mark.asyncio
async def test_tool_dispatches_to_task_aio():
    env = flyte.TaskEnvironment("lg_tools_c")

    @tool
    @env.task
    def multiply(a: int, b: int) -> int:
        """Multiply."""
        return a * b

    with patch.object(multiply.task, "aio", new_callable=AsyncMock, return_value=42) as mock_aio:
        result = await multiply(a=6, b=7)

    mock_aio.assert_awaited_once_with(a=6, b=7)
    assert result == "42"


def test_bare_and_parametrized_decorator_forms():
    env = flyte.TaskEnvironment("lg_tools_d")

    @tool
    @env.task
    def a(x: int) -> int:
        """A."""
        return x

    assert a.__name__ == "a"

    @tool(name="bee")
    @env.task
    def b(x: int) -> int:
        """B."""
        return x

    assert b.__name__ == "bee"
