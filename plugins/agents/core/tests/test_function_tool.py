"""Tests for the shared generic ``function_tool`` (plain-callable tool wrapper).

This lives in core because the Mistral and Google ADK adapters share it verbatim
(their SDKs accept plain Python callables as tools). OpenAI/Claude provide their own
SDK-native versions, so they are exercised in their own packages.
"""

import inspect
from unittest.mock import AsyncMock, patch

import flyte
import pytest

from flyteplugins.agents.core import ToolTaskResolver, function_tool


def test_task_becomes_plain_tool_with_resolver():
    env = flyte.TaskEnvironment("core_ft_a")

    @function_tool
    @env.task
    def get_weather(city: str) -> str:
        """Get weather."""
        return city

    assert inspect.isfunction(get_weather)
    assert get_weather.__name__ == "get_weather"
    assert get_weather.__wrapped_task__ is get_weather.task
    assert isinstance(get_weather.task.task_resolver, ToolTaskResolver)


def test_wrapper_preserves_the_task_signature():
    env = flyte.TaskEnvironment("core_ft_b")

    @function_tool
    @env.task
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    # functools.wraps -> the SDK derives the right declaration from the real params.
    assert list(inspect.signature(add).parameters) == ["a", "b"]


def test_plain_callable_passes_through():
    def f(x: int) -> int:
        return x

    assert function_tool(f) is f


@pytest.mark.asyncio
async def test_tool_dispatches_to_task_aio():
    env = flyte.TaskEnvironment("core_ft_c")

    @function_tool
    @env.task
    def multiply(a: int, b: int) -> int:
        """Multiply."""
        return a * b

    with patch.object(multiply.task, "aio", new_callable=AsyncMock, return_value=42) as mock_aio:
        result = await multiply(a=6, b=7)

    mock_aio.assert_awaited_once_with(a=6, b=7)
    assert result == 42
