"""Unit tests for the DeepSeek adapter's tool wrapper (no network / no controller)."""

import re
from unittest.mock import AsyncMock, patch

import flyte
import pytest
from flyteplugins.agents.core import ToolTaskResolver

from flyteplugins.agents.deepseek import HarnessTool, tool


def test_task_becomes_harness_tool_with_resolver():
    env = flyte.TaskEnvironment("ds_tools_a")

    @tool
    @env.task
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"sunny in {city}"

    assert isinstance(get_weather, HarnessTool)
    assert get_weather.name == "get_weather"
    assert get_weather.description == "Get the current weather for a city."
    # The tool shadows the task at module scope, so the real task stays reachable
    # and resolves to itself on the worker instead of re-dispatching the tool.
    assert get_weather.__wrapped_task__ is get_weather.task
    assert isinstance(get_weather.task.task_resolver, ToolTaskResolver)


def test_schema_comes_from_the_flyte_type_engine():
    env = flyte.TaskEnvironment("ds_tools_b")

    @tool
    @env.task
    def book(city: str, nights: int, budget: float, refundable: bool = True) -> str:
        """Book a stay."""
        return city

    properties = book.schema["properties"]
    assert properties["city"]["type"] == "string"
    assert properties["nights"]["type"] == "integer"
    assert properties["budget"]["type"] == "number"
    assert properties["refundable"]["type"] == "boolean"


def test_name_and_description_overrides():
    env = flyte.TaskEnvironment("ds_tools_c")

    @tool(name="weather", description="Weather lookup.")
    @env.task
    def get_weather(city: str) -> str:
        """Original docstring."""
        return city

    assert get_weather.name == "weather"
    assert get_weather.description == "Weather lookup."


def test_usage_line_names_params_and_marks_optionals():
    env = flyte.TaskEnvironment("ds_tools_d")

    @tool
    @env.task
    def search(query: str, limit: int = 5) -> str:
        """Search the index."""
        return query

    usage = search.usage()
    assert usage.startswith("search(")
    assert "query: string" in usage
    assert "limit: integer (optional)" in usage
    assert usage.endswith("— Search the index.")


@pytest.mark.asyncio
async def test_invoke_dispatches_to_task_aio():
    """``invoke`` goes through ``task.aio`` — the durable child-action seam."""
    env = flyte.TaskEnvironment("ds_tools_e")

    @tool
    @env.task
    async def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"sunny in {city}"

    with patch.object(get_weather.task, "aio", new_callable=AsyncMock, return_value="sunny in Paris") as mock_aio:
        assert await get_weather.invoke({"city": "Paris"}) == "sunny in Paris"

    mock_aio.assert_awaited_once_with(city="Paris")


@pytest.mark.asyncio
async def test_invoke_coerces_llm_ints_to_floats():
    """An LLM emits ``42`` for a float param; the type engine would reject it."""
    env = flyte.TaskEnvironment("ds_tools_f")

    @tool
    @env.task
    async def convert(amount_usd: float) -> float:
        """Convert dollars to euros."""
        return amount_usd * 0.9

    with patch.object(convert.task, "aio", new_callable=AsyncMock, return_value=37.8) as mock_aio:
        assert await convert.invoke({"amount_usd": 42}) == "37.8"

    mock_aio.assert_awaited_once_with(amount_usd=42.0)
    assert isinstance(mock_aio.await_args.kwargs["amount_usd"], float)


@pytest.mark.asyncio
async def test_non_string_results_are_json_encoded():
    env = flyte.TaskEnvironment("ds_tools_g")

    @tool
    @env.task
    async def stats(city: str) -> dict:
        """Get city stats."""
        return {"city": city, "population": 2102650}

    payload = {"city": "Paris", "population": 2102650}
    with patch.object(stats.task, "aio", new_callable=AsyncMock, return_value=payload):
        assert await stats.invoke({"city": "Paris"}) == '{"city": "Paris", "population": 2102650}'


@pytest.mark.asyncio
async def test_plain_callable_becomes_a_tool():
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    add_tool = tool(add)
    assert isinstance(add_tool, HarnessTool)
    assert add_tool.name == "add"
    assert await add_tool.invoke({"a": 2, "b": 3}) == "5"


def test_tool_rejects_non_callables():
    with pytest.raises(TypeError, match=re.escape("expects a Flyte @env.task or a callable")):
        tool(42)
