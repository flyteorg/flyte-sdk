"""Unit tests for Anthropic agents plugin."""

import flyte
import pytest

from flyteplugins.anthropic import Agent, function_tool
from flyteplugins.anthropic.agents._function_tools import (
    FunctionTool,
    _get_function_schema,
    _python_type_to_json_schema,
)

# -- Type conversion tests --


def test_python_type_to_json_schema_string():
    schema = _python_type_to_json_schema(str)
    assert schema == {"type": "string"}


def test_python_type_to_json_schema_int():
    schema = _python_type_to_json_schema(int)
    assert schema == {"type": "integer"}


def test_python_type_to_json_schema_float():
    schema = _python_type_to_json_schema(float)
    assert schema == {"type": "number"}


def test_python_type_to_json_schema_bool():
    schema = _python_type_to_json_schema(bool)
    assert schema == {"type": "boolean"}


def test_python_type_to_json_schema_list():
    schema = _python_type_to_json_schema(list[str])
    assert schema == {"type": "array", "items": {"type": "string"}}


def test_python_type_to_json_schema_dict():
    schema = _python_type_to_json_schema(dict[str, int])
    assert schema == {"type": "object"}


def test_python_type_to_json_schema_optional():
    from typing import Optional

    schema = _python_type_to_json_schema(Optional[str])
    assert schema == {"type": "string"}


# -- Function schema tests --


def test_get_function_schema_simple():
    def my_func(name: str, age: int) -> str:
        return f"{name} is {age}"

    schema = _get_function_schema(my_func)
    assert schema["type"] == "object"
    assert "name" in schema["properties"]
    assert "age" in schema["properties"]
    assert schema["properties"]["name"] == {"type": "string"}
    assert schema["properties"]["age"] == {"type": "integer"}
    assert set(schema["required"]) == {"name", "age"}


def test_get_function_schema_with_defaults():
    def my_func(name: str, age: int = 25) -> str:
        return f"{name} is {age}"

    schema = _get_function_schema(my_func)
    assert schema["required"] == ["name"]


# -- function_tool tests --


def test_function_tool_with_regular_function():
    def my_function(prompt: str) -> str:
        """Process the prompt."""
        return f"Hello, {prompt}!"

    tool = function_tool(my_function)
    assert isinstance(tool, FunctionTool)
    assert tool.name == "my_function"
    assert tool.description == "Process the prompt."
    assert "prompt" in tool.input_schema["properties"]


def test_function_tool_with_custom_name():
    def my_function(prompt: str) -> str:
        """Process the prompt."""
        return f"Hello, {prompt}!"

    tool = function_tool(my_function, name="custom_name")
    assert tool.name == "custom_name"


def test_function_tool_with_custom_description():
    def my_function(prompt: str) -> str:
        return f"Hello, {prompt}!"

    tool = function_tool(my_function, description="Custom description")
    assert tool.description == "Custom description"


def test_function_tool_with_flyte_task():
    env = flyte.TaskEnvironment("test")

    @env.task
    def my_task(prompt: str) -> str:
        """A flyte task."""
        return f"Hello, {prompt}!"

    tool = function_tool(my_task)
    assert isinstance(tool, FunctionTool)
    assert tool.task is my_task
    assert tool.name == "my_task"


def test_function_tool_with_async_function():
    async def my_async_function(prompt: str) -> str:
        """Async function."""
        return f"Hello, {prompt}!"

    tool = function_tool(my_async_function)
    assert tool.is_async is True


def test_function_tool_to_anthropic_format():
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"Weather in {city}"

    tool = function_tool(get_weather)
    anthropic_tool = tool.to_anthropic_tool()

    assert anthropic_tool["name"] == "get_weather"
    assert anthropic_tool["description"] == "Get the weather for a city."
    assert "input_schema" in anthropic_tool
    assert anthropic_tool["input_schema"]["properties"]["city"] == {"type": "string"}


# -- Agent tests --


def test_agent_creation():
    agent = Agent(
        name="test_agent",
        instructions="You are a test agent.",
        model="claude-sonnet-4-20250514",
    )
    assert agent.name == "test_agent"
    assert agent.instructions == "You are a test agent."


def test_agent_with_tools():
    def my_tool(x: int) -> int:
        """Double the number."""
        return x * 2

    tool = function_tool(my_tool)
    agent = Agent(name="test", tools=[tool])

    anthropic_tools = agent.get_anthropic_tools()
    assert len(anthropic_tools) == 1
    assert anthropic_tools[0]["name"] == "my_tool"


# -- Execute tests --


@pytest.mark.asyncio
async def test_function_tool_execute_sync():
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    tool = function_tool(add)
    result = await tool.execute(a=2, b=3)
    assert result == 5


@pytest.mark.asyncio
async def test_function_tool_execute_async():
    async def add_async(a: int, b: int) -> int:
        """Add two numbers asynchronously."""
        return a + b

    tool = function_tool(add_async)
    result = await tool.execute(a=2, b=3)
    assert result == 5
