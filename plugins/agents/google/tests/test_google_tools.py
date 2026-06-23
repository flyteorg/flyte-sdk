"""Unit tests for the Google ADK adapter's tool bridge (no network)."""

import inspect
from unittest.mock import AsyncMock, patch

import flyte
import pytest
from flyteplugins.agents.core import ToolTaskResolver

from flyteplugins.agents.google import function_tool


def test_task_becomes_adk_tool_with_resolver():
    env = flyte.TaskEnvironment("g_tools_a")

    @function_tool
    @env.task
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"sunny in {city}"

    # A plain (ADK-usable) function carrying the task + resolver.
    assert inspect.isfunction(get_weather)
    assert get_weather.__name__ == "get_weather"
    assert get_weather.__wrapped_task__ is get_weather.task
    assert isinstance(get_weather.task.task_resolver, ToolTaskResolver)


def test_adk_agent_accepts_the_tool():
    from google.adk.agents import LlmAgent

    env = flyte.TaskEnvironment("g_tools_b")

    @function_tool
    @env.task
    def get_weather(city: str) -> str:
        """Get weather."""
        return city

    agent = LlmAgent(name="a", model="gemini-2.0-flash", instruction="hi", tools=[get_weather])
    assert len(agent.tools) == 1


@pytest.mark.asyncio
async def test_tool_dispatches_to_task_aio():
    env = flyte.TaskEnvironment("g_tools_c")

    @function_tool
    @env.task
    def multiply(a: int, b: int) -> int:
        """Multiply."""
        return a * b

    with patch.object(multiply.task, "aio", new_callable=AsyncMock, return_value=42) as mock_aio:
        result = await multiply(a=6, b=7)

    mock_aio.assert_awaited_once_with(a=6, b=7)
    assert result == 42
