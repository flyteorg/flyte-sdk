"""Unit tests for the Hermes adapter's tool bridge (no network / no controller)."""

import inspect
from unittest.mock import AsyncMock, patch

import flyte
import pytest
from flyteplugins.agents.core import ToolTaskResolver

from flyteplugins.agents.hermes import FLYTE_TOOLSET, tool


def test_task_becomes_hermes_tool_with_resolver():
    env = flyte.TaskEnvironment("h_tools_a")

    @tool
    @env.task
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"sunny in {city}"

    # A plain function carrying the task + resolver (the shared core wrapper).
    assert inspect.isfunction(get_weather)
    assert get_weather.__name__ == "get_weather"
    assert get_weather.__wrapped_task__ is get_weather.task
    assert isinstance(get_weather.task.task_resolver, ToolTaskResolver)
    assert get_weather.__hermes_registered__ is True


def test_tool_registers_in_the_hermes_registry():
    """The real API: tools live in hermes' global registry, under FLYTE_TOOLSET."""
    from tools.registry import registry  # hermes-agent

    env = flyte.TaskEnvironment("h_tools_b")

    @tool
    @env.task
    def get_tides(city: str) -> str:
        """Get the tide report for a city."""
        return f"low tide in {city}"

    entry = registry.get_entry("get_tides")
    assert entry is not None
    assert entry.toolset == FLYTE_TOOLSET
    assert entry.is_async is True
    assert entry.schema["function"]["name"] == "get_tides"
    assert entry.schema["function"]["parameters"]["properties"] == {"city": {"type": "string"}}


def test_hermes_agent_exposes_the_tool():
    """A real AIAgent built with enabled_toolsets=[FLYTE_TOOLSET] sees the tool."""
    try:
        from run_agent import AIAgent  # hermes-agent's top-level module
    except ModuleNotFoundError:
        pytest.skip("hermes-agent not installed")

    env = flyte.TaskEnvironment("h_tools_c")

    @tool
    @env.task
    def get_sunshine(city: str) -> str:
        """Get sunshine hours."""
        return city

    # Offline: an explicit (fake) api_key/base_url skips hermes' provider-config
    # resolution; construction makes no network calls.
    agent = AIAgent(
        model="test-model",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        quiet_mode=True,
        enabled_toolsets=[FLYTE_TOOLSET],
    )
    assert "get_sunshine" in [t["function"]["name"] for t in agent.tools]


@pytest.mark.asyncio
async def test_tool_dispatches_to_task_aio():
    env = flyte.TaskEnvironment("h_tools_d")

    @tool
    @env.task
    def multiply(a: int, b: int) -> int:
        """Multiply."""
        return a * b

    with patch.object(multiply.task, "aio", new_callable=AsyncMock, return_value=42) as mock_aio:
        result = await multiply(a=6, b=7)

    mock_aio.assert_awaited_once_with(a=6, b=7)
    assert result == 42


def test_hermes_dispatch_runs_the_flyte_task():
    """End-to-end through hermes' own dispatch: registry -> handler -> task.aio."""
    from tools.registry import registry  # hermes-agent

    env = flyte.TaskEnvironment("h_tools_e")

    @tool
    @env.task
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    with patch.object(add.task, "aio", new_callable=AsyncMock, return_value=13) as mock_aio:
        out = registry.dispatch("add", {"a": 6, "b": 7})

    mock_aio.assert_awaited_once_with(a=6, b=7)
    assert out == "13"


def test_bare_and_parametrized_decorator_forms():
    env = flyte.TaskEnvironment("h_tools_f")

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
