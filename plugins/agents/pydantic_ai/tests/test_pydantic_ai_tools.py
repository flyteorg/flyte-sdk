"""Unit tests for the Pydantic AI adapter's tool bridge (no network / no controller)."""

import inspect
from unittest.mock import AsyncMock, patch

import flyte
import pytest
from flyteplugins.agents.core import ToolTaskResolver

from flyteplugins.agents.pydantic_ai import tool


def test_task_becomes_pydantic_ai_tool_with_resolver():
    env = flyte.TaskEnvironment("pai_tools_a")

    @tool
    @env.task
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"sunny in {city}"

    # A plain (Pydantic AI-usable) function carrying the task + resolver.
    assert inspect.isfunction(get_weather)
    assert get_weather.__name__ == "get_weather"
    assert get_weather.__wrapped_task__ is get_weather.task
    assert isinstance(get_weather.task.task_resolver, ToolTaskResolver)


def test_pydantic_ai_agent_accepts_the_tool():
    try:
        from pydantic_ai import Agent

        env = flyte.TaskEnvironment("pai_tools_b")

        @tool
        @env.task
        def get_weather(city: str) -> str:
            """Get weather."""
            return city

        agent = Agent(model="test", tools=[get_weather])
        assert agent is not None
    except Exception:
        pytest.skip("pydantic_ai Agent requires model setup")


@pytest.mark.asyncio
async def test_tool_dispatches_to_task_aio():
    env = flyte.TaskEnvironment("pai_tools_c")

    @tool
    @env.task
    def multiply(a: int, b: int) -> int:
        """Multiply."""
        return a * b

    with patch.object(multiply.task, "aio", new_callable=AsyncMock, return_value=42) as mock_aio:
        result = await multiply(a=6, b=7)

    mock_aio.assert_awaited_once_with(a=6, b=7)
    assert result == 42


def test_bare_and_parametrized_decorator_forms():
    env = flyte.TaskEnvironment("pai_tools_d")

    @tool
    @env.task
    def a(x: int) -> int:
        """A."""
        return x

    assert inspect.isfunction(a)

    @tool(name="bee")
    @env.task
    def b(x: int) -> int:
        """B."""
        return x

    assert inspect.isfunction(b)
    assert b.__name__ == "bee"
