"""Unit tests for the CrewAI adapter's tool bridge (no network / no controller)."""

from unittest.mock import AsyncMock, patch

import flyte
import pytest
from crewai.tools import BaseTool
from flyteplugins.agents.core import ToolTaskResolver

from flyteplugins.agents.crewai import tool


def test_task_becomes_crewai_basetool_with_resolver():
    env = flyte.TaskEnvironment("crewai_tools_a")

    @tool
    @env.task
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"sunny in {city}"

    # The tool is a native CrewAI BaseTool carrying the task + resolver.
    assert isinstance(get_weather, BaseTool)
    assert get_weather.name == "get_weather"
    assert get_weather.__wrapped_task__ is get_weather.task
    assert isinstance(get_weather.task.task_resolver, ToolTaskResolver)
    # The args schema is derived from the task's parameters.
    assert list(get_weather.args_schema.model_fields.keys()) == ["city"]


def test_crewai_agent_accepts_the_tool():
    from crewai import Agent

    env = flyte.TaskEnvironment("crewai_tools_b")

    @tool
    @env.task
    def get_weather(city: str) -> str:
        """Get weather."""
        return city

    agent = Agent(
        role="Assistant",
        goal="Help the user.",
        backstory="You are helpful.",
        tools=[get_weather],
        llm="gpt-4o",
    )
    assert len(agent.tools) == 1
    assert all(isinstance(t, BaseTool) for t in agent.tools)


@pytest.mark.asyncio
async def test_tool_dispatches_to_task_aio_async():
    env = flyte.TaskEnvironment("crewai_tools_c")

    @tool
    @env.task
    def multiply(a: int, b: int) -> int:
        """Multiply."""
        return a * b

    # CrewAI's native async path: _arun awaits task.aio directly.
    with patch.object(multiply.task, "aio", new_callable=AsyncMock, return_value=42) as mock_aio:
        result = await multiply._arun(a=6, b=7)

    mock_aio.assert_awaited_once_with(a=6, b=7)
    assert result == "42"


def test_tool_sync_run_bridges_to_task_aio():
    """CrewAI invokes tools synchronously; the sync ``_run`` must reach ``task.aio``."""
    env = flyte.TaskEnvironment("crewai_tools_d")

    @tool
    @env.task
    def multiply(a: int, b: int) -> int:
        """Multiply."""
        return a * b

    with patch.object(multiply.task, "aio", new_callable=AsyncMock, return_value=99) as mock_aio:
        result = multiply._run(a=3, b=3)

    mock_aio.assert_awaited_once_with(a=3, b=3)
    assert result == "99"


def test_structured_tool_invoke_path():
    """The path CrewAI actually drives during an agent loop (invoke -> _run)."""
    env = flyte.TaskEnvironment("crewai_tools_e")

    @tool
    @env.task
    def get_weather(city: str) -> str:
        """Get weather."""
        return f"sunny in {city}"

    structured = get_weather.to_structured_tool()
    assert structured.invoke({"city": "Paris"}) == "sunny in Paris"


def test_bare_and_parametrized_decorator_forms():
    env = flyte.TaskEnvironment("crewai_tools_f")

    @tool
    @env.task
    def a(x: int) -> int:
        """A."""
        return x

    assert a.name == "a"

    @tool(name="bee")
    @env.task
    def b(x: int) -> int:
        """B."""
        return x

    assert b.name == "bee"


def test_plain_callable_becomes_basetool():
    def greet(name: str) -> str:
        """Greet someone."""
        return f"hi {name}"

    t = tool(greet)
    assert isinstance(t, BaseTool)
    assert t.name == "greet"
    assert t.__wrapped_task__ is None
    assert t.to_structured_tool().invoke({"name": "Ann"}) == "hi Ann"
