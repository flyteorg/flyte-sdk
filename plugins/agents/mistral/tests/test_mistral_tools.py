"""Unit tests for the Mistral adapter's tool bridge (no network)."""

import inspect
from unittest.mock import AsyncMock, patch

import flyte
import pytest
from flyteplugins.agents.core import ToolTaskResolver

from flyteplugins.agents.mistral import function_tool


def test_task_becomes_registerable_tool_with_resolver():
    env = flyte.TaskEnvironment("mistral_tools_a")

    @function_tool
    @env.task
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"sunny in {city}"

    # It's a plain (registerable) function carrying the task + resolver.
    assert inspect.isfunction(get_weather)
    assert get_weather.__name__ == "get_weather"
    assert get_weather.__wrapped_task__ is get_weather.task
    assert isinstance(get_weather.task.task_resolver, ToolTaskResolver)


def test_tool_schema_is_derived_from_the_task_signature():
    from mistralai.extra.run.tools import create_tool_call

    env = flyte.TaskEnvironment("mistral_tools_b")

    @function_tool
    @env.task
    def get_weather(city: str) -> str:
        """Get weather."""
        return city

    spec = create_tool_call(get_weather)
    assert spec.function.name == "get_weather"
    assert "city" in spec.function.parameters["properties"]


@pytest.mark.asyncio
async def test_tool_dispatches_to_task_aio():
    env = flyte.TaskEnvironment("mistral_tools_c")

    @function_tool
    @env.task
    def multiply(a: int, b: int) -> int:
        """Multiply."""
        return a * b

    with patch.object(multiply.task, "aio", new_callable=AsyncMock, return_value=42) as mock_aio:
        result = await multiply(a=6, b=7)

    mock_aio.assert_awaited_once_with(a=6, b=7)
    assert result == 42
