"""Unit tests for the Claude adapter's tool bridge (no CLI / no network)."""

from unittest.mock import AsyncMock, patch

import flyte
import pytest
from claude_agent_sdk import SdkMcpTool
from flyteplugins.agents.core import ToolTaskResolver

from flyteplugins.agents.claude import tool


def test_task_becomes_sdk_mcp_tool_with_resolver():
    env = flyte.TaskEnvironment("claude_tools_a")

    @tool
    @env.task
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"sunny in {city}"

    assert isinstance(get_weather, SdkMcpTool)
    assert get_weather.name == "get_weather"
    assert get_weather.description == "Get the current weather for a city."
    # input schema comes from the Flyte type engine
    assert "city" in get_weather.input_schema.get("properties", {})
    # the real task is recoverable on the worker, and wired to the resolver
    assert get_weather.__wrapped_task__ is get_weather.task
    assert isinstance(get_weather.task.task_resolver, ToolTaskResolver)


@pytest.mark.asyncio
async def test_tool_handler_dispatches_to_task_aio():
    env = flyte.TaskEnvironment("claude_tools_b")

    @tool
    @env.task
    def echo(text: str) -> str:
        """Echo."""
        return text

    with patch.object(echo.task, "aio", new_callable=AsyncMock, return_value="hello") as mock_aio:
        out = await echo.handler({"text": "hi"})

    mock_aio.assert_awaited_once_with(text="hi")
    assert out == {"content": [{"type": "text", "text": "hello"}]}


@pytest.mark.asyncio
async def test_tool_handler_serializes_non_string_result():
    env = flyte.TaskEnvironment("claude_tools_c")

    @tool
    @env.task
    def get_data() -> dict:
        """Get data."""
        return {"temp": 72}

    with patch.object(get_data.task, "aio", new_callable=AsyncMock, return_value={"temp": 72}):
        out = await get_data.handler({})

    assert '"temp"' in out["content"][0]["text"]
