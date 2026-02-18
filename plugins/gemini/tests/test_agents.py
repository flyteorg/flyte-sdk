"""Unit tests for Gemini agents plugin."""

from typing import Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

import flyte
import pytest

from flyteplugins.gemini import Agent, function_tool, run_agent
from flyteplugins.gemini.agents._function_tools import (
    FunctionTool,
    _get_function_schema,
    _python_type_to_json_schema,
)

# ---------------------------------------------------------------------------
# Type conversion tests
# ---------------------------------------------------------------------------


def test_python_type_to_json_schema_string():
    assert _python_type_to_json_schema(str) == {"type": "string"}


def test_python_type_to_json_schema_int():
    assert _python_type_to_json_schema(int) == {"type": "integer"}


def test_python_type_to_json_schema_float():
    assert _python_type_to_json_schema(float) == {"type": "number"}


def test_python_type_to_json_schema_bool():
    assert _python_type_to_json_schema(bool) == {"type": "boolean"}


def test_python_type_to_json_schema_list():
    assert _python_type_to_json_schema(list[str]) == {"type": "array", "items": {"type": "string"}}


def test_python_type_to_json_schema_bare_list():
    assert _python_type_to_json_schema(list) == {"type": "array"}


def test_python_type_to_json_schema_dict():
    assert _python_type_to_json_schema(dict[str, int]) == {"type": "object"}


def test_python_type_to_json_schema_bare_dict():
    assert _python_type_to_json_schema(dict) == {"type": "object"}


def test_python_type_to_json_schema_optional():
    assert _python_type_to_json_schema(Optional[str]) == {"type": "string"}


def test_python_type_to_json_schema_nested_list():
    schema = _python_type_to_json_schema(list[list[int]])
    assert schema == {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}


def test_python_type_to_json_schema_optional_list():
    schema = _python_type_to_json_schema(Optional[list[str]])
    assert schema == {"type": "array", "items": {"type": "string"}}


def test_python_type_to_json_schema_union():
    schema = _python_type_to_json_schema(Union[str, int])
    assert schema == {"anyOf": [{"type": "string"}, {"type": "integer"}]}


def test_python_type_to_json_schema_unknown_type():
    class MyCustomClass:
        pass

    assert _python_type_to_json_schema(MyCustomClass) == {"type": "string"}


# ---------------------------------------------------------------------------
# Function schema tests
# ---------------------------------------------------------------------------


def test_get_function_schema_simple():
    def my_func(name: str, age: int) -> str:
        return f"{name} is {age}"

    schema = _get_function_schema(my_func)
    assert schema["type"] == "object"
    assert schema["properties"]["name"] == {"type": "string"}
    assert schema["properties"]["age"] == {"type": "integer"}
    assert set(schema["required"]) == {"name", "age"}


def test_get_function_schema_with_defaults():
    def my_func(name: str, age: int = 25) -> str:
        return f"{name} is {age}"

    schema = _get_function_schema(my_func)
    assert schema["required"] == ["name"]


def test_get_function_schema_no_type_hints():
    def my_func(name, age):
        return f"{name} is {age}"

    schema = _get_function_schema(my_func)
    # Untyped parameters default to string
    assert schema["properties"]["name"] == {"type": "string"}
    assert schema["properties"]["age"] == {"type": "string"}


def test_get_function_schema_all_optional():
    def my_func(x: int = 0, y: int = 0) -> int:
        return x + y

    schema = _get_function_schema(my_func)
    assert schema["required"] == []


def test_get_function_schema_skips_self_cls():
    def my_method(self, name: str) -> str:
        return name

    schema = _get_function_schema(my_method)
    assert "self" not in schema["properties"]
    assert "name" in schema["properties"]


# ---------------------------------------------------------------------------
# function_tool tests
# ---------------------------------------------------------------------------


def test_function_tool_with_regular_function():
    def my_function(prompt: str) -> str:
        """Process the prompt."""
        return f"Hello, {prompt}!"

    tool = function_tool(my_function)
    assert isinstance(tool, FunctionTool)
    assert tool.name == "my_function"
    assert tool.description == "Process the prompt."
    assert "prompt" in tool.input_schema["properties"]
    assert tool.task is None
    assert tool.native_interface is None
    assert tool.report is False


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


def test_function_tool_without_docstring():
    def no_docs(x: int) -> int:
        return x * 2

    tool = function_tool(no_docs)
    assert tool.description == "Execute no_docs"


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
    assert tool.native_interface is my_task.interface
    assert tool.report == my_task.report


def test_function_tool_with_async_function():
    async def my_async_function(prompt: str) -> str:
        """Async function."""
        return f"Hello, {prompt}!"

    tool = function_tool(my_async_function)
    assert tool.is_async is True


def test_function_tool_sync_not_async():
    def my_sync_function(prompt: str) -> str:
        """Sync function."""
        return f"Hello, {prompt}!"

    tool = function_tool(my_sync_function)
    assert tool.is_async is False


def test_function_tool_to_gemini_format():
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"Weather in {city}"

    tool = function_tool(get_weather)
    gemini_tool = tool.to_gemini_tool()

    assert gemini_tool.name == "get_weather"
    assert gemini_tool.description == "Get the weather for a city."
    assert gemini_tool.parameters_json_schema is not None
    assert gemini_tool.parameters_json_schema["properties"]["city"] == {"type": "string"}


def test_function_tool_as_decorator():
    tool = function_tool(name="custom", description="A custom tool")

    @tool
    def my_func(x: int) -> int:
        """My func."""
        return x

    assert isinstance(my_func, FunctionTool)
    assert my_func.name == "custom"
    assert my_func.description == "A custom tool"


def test_function_tool_with_flyte_trace():
    @flyte.trace
    async def traced_function(prompt: str) -> str:
        """A traced function."""
        return f"traced: {prompt}"

    tool = function_tool(traced_function)
    assert isinstance(tool, FunctionTool)
    assert tool.name == "traced_function"
    assert tool.description == "A traced function."
    assert tool.task is None
    assert tool.is_async is True


def test_function_tool_name_preserved():
    """Verify that two tools can be created with different names for the same function."""

    def helper(x: int) -> int:
        """Helper."""
        return x

    tool_a = function_tool(helper, name="tool_a")
    tool_b = function_tool(helper, name="tool_b")
    assert tool_a.name == "tool_a"
    assert tool_b.name == "tool_b"


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------


def test_agent_creation():
    agent = Agent(
        name="test_agent",
        instructions="You are a test agent.",
        model="gemini-2.5-flash",
    )
    assert agent.name == "test_agent"
    assert agent.instructions == "You are a test agent."
    assert agent.max_output_tokens == 8192
    assert agent.max_iterations == 10


def test_agent_defaults():
    agent = Agent()
    assert agent.name == "assistant"
    assert agent.instructions == "You are a helpful assistant."
    assert agent.model == "gemini-2.5-flash"
    assert agent.tools == []


def test_agent_with_tools():
    def my_tool(x: int) -> int:
        """Double the number."""
        return x * 2

    tool = function_tool(my_tool)
    agent = Agent(name="test", tools=[tool])

    gemini_tools = agent.get_gemini_tools()
    assert len(gemini_tools) == 1
    assert gemini_tools[0].name == "my_tool"


# ---------------------------------------------------------------------------
# Execute tests
# ---------------------------------------------------------------------------


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


@pytest.mark.asyncio
async def test_function_tool_execute_flyte_task():
    env = flyte.TaskEnvironment("test-exec")

    @env.task
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    tool = function_tool(multiply)
    result = await tool.execute(a=3, b=4)
    assert result == 12


# ---------------------------------------------------------------------------
# Helpers for run_agent mock tests
# ---------------------------------------------------------------------------


def _make_text_part(text: str):
    """Create a mock text part."""
    part = MagicMock()
    part.text = text
    part.function_call = None
    return part


def _make_function_call_part(name: str, args: dict):
    """Create a mock function_call part."""
    part = MagicMock()
    part.text = None
    part.function_call = MagicMock()
    part.function_call.name = name
    part.function_call.args = args
    return part


def _make_response(finish_reason: str, parts: list):
    """Create a mock Gemini API response."""
    content = MagicMock()
    content.parts = parts

    candidate = MagicMock()
    candidate.finish_reason = finish_reason
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    return response


# ---------------------------------------------------------------------------
# run_agent tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_agent_missing_api_key():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="Google API key is required"):
            await run_agent(prompt="hello", api_key=None)


@pytest.mark.asyncio
async def test_run_agent_simple_text_response():
    mock_response = _make_response("STOP", [_make_text_part("Hello!")])

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch("flyteplugins.gemini.agents._function_tools.genai.Client", return_value=mock_client):
        result = await run_agent(prompt="Hi", api_key="test-key")

    assert result == "Hello!"


@pytest.mark.asyncio
async def test_run_agent_empty_text_response():
    mock_response = _make_response("STOP", [_make_text_part("")])

    # Part with empty text and no function_call should still count as text
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch("flyteplugins.gemini.agents._function_tools.genai.Client", return_value=mock_client):
        result = await run_agent(prompt="Hi", api_key="test-key")

    assert result == ""


@pytest.mark.asyncio
async def test_run_agent_with_tool_call():
    """Test the full tool call loop: function_call -> text response."""

    def get_weather(city: str) -> str:
        """Get weather."""
        return f"Sunny in {city}"

    tool = function_tool(get_weather)

    # First response: Gemini wants to call a function
    tool_response = _make_response(
        "STOP",
        [_make_function_call_part("get_weather", {"city": "SF"})],
    )

    # Second response: Gemini gives final answer
    final_response = _make_response("STOP", [_make_text_part("It's sunny in SF!")])

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=[tool_response, final_response])

    with patch("flyteplugins.gemini.agents._function_tools.genai.Client", return_value=mock_client):
        result = await run_agent(prompt="Weather in SF?", tools=[tool], api_key="test-key")

    assert result == "It's sunny in SF!"
    assert mock_client.aio.models.generate_content.call_count == 2


@pytest.mark.asyncio
async def test_run_agent_unknown_tool():
    """Test that unknown tool names are handled gracefully with error response."""

    tool_response = _make_response(
        "STOP",
        [_make_function_call_part("nonexistent_tool", {})],
    )
    final_response = _make_response("STOP", [_make_text_part("Sorry, tool not found.")])

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=[tool_response, final_response])

    with patch("flyteplugins.gemini.agents._function_tools.genai.Client", return_value=mock_client):
        result = await run_agent(prompt="test", tools=[], api_key="test-key")

    assert result == "Sorry, tool not found."


@pytest.mark.asyncio
async def test_run_agent_tool_execution_error():
    """Test that tool execution errors are caught and sent back to Gemini."""

    def failing_tool(x: int) -> str:
        """Always fails."""
        raise RuntimeError("Something broke")

    tool = function_tool(failing_tool)

    tool_response = _make_response(
        "STOP",
        [_make_function_call_part("failing_tool", {"x": 1})],
    )
    final_response = _make_response("STOP", [_make_text_part("Tool failed, sorry.")])

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=[tool_response, final_response])

    with patch("flyteplugins.gemini.agents._function_tools.genai.Client", return_value=mock_client):
        result = await run_agent(prompt="test", tools=[tool], api_key="test-key")

    assert result == "Tool failed, sorry."


@pytest.mark.asyncio
async def test_run_agent_max_iterations():
    """Test that the agent stops after max_iterations."""

    def dummy(x: int) -> str:
        """Dummy."""
        return str(x)

    tool = function_tool(dummy)

    # Always returns function_call, never ends
    infinite_tool_response = _make_response(
        "STOP",
        [_make_function_call_part("dummy", {"x": 1})],
    )

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=infinite_tool_response)

    with patch("flyteplugins.gemini.agents._function_tools.genai.Client", return_value=mock_client):
        result = await run_agent(prompt="test", tools=[tool], api_key="test-key", max_iterations=3)

    assert result == "Maximum iterations reached without final response."
    assert mock_client.aio.models.generate_content.call_count == 3


@pytest.mark.asyncio
async def test_run_agent_max_tokens_finish_reason():
    """Test that MAX_TOKENS finish_reason returns partial text."""
    mock_response = _make_response("MAX_TOKENS", [_make_text_part("Partial response that was cut")])

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch("flyteplugins.gemini.agents._function_tools.genai.Client", return_value=mock_client):
        result = await run_agent(prompt="test", api_key="test-key")

    assert result == "Partial response that was cut"


@pytest.mark.asyncio
async def test_run_agent_safety_finish_reason():
    """Test that SAFETY finish_reason returns informative message."""
    # SAFETY block with no text parts
    part = MagicMock()
    part.text = None
    part.function_call = None
    mock_response = _make_response("SAFETY", [part])

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch("flyteplugins.gemini.agents._function_tools.genai.Client", return_value=mock_client):
        result = await run_agent(prompt="test", api_key="test-key")

    assert "safety" in result.lower()


@pytest.mark.asyncio
async def test_run_agent_unexpected_finish_reason():
    """Test that unexpected finish_reason with no text returns informative message."""
    part = MagicMock()
    part.text = None
    part.function_call = None
    mock_response = _make_response("RECITATION", [part])

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch("flyteplugins.gemini.agents._function_tools.genai.Client", return_value=mock_client):
        result = await run_agent(prompt="test", api_key="test-key")

    assert "RECITATION" in result


@pytest.mark.asyncio
async def test_run_agent_with_agent_config():
    """Test that Agent config overrides default parameters."""

    def my_tool(x: int) -> int:
        """Double."""
        return x * 2

    tool = function_tool(my_tool)
    agent = Agent(
        name="test",
        instructions="Be precise.",
        model="gemini-2.0-flash",
        tools=[tool],
        max_output_tokens=1024,
        max_iterations=5,
    )

    mock_response = _make_response("STOP", [_make_text_part("Done")])

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch("flyteplugins.gemini.agents._function_tools.genai.Client", return_value=mock_client):
        result = await run_agent(prompt="test", agent=agent, api_key="test-key")

    assert result == "Done"
    call_kwargs = mock_client.aio.models.generate_content.call_args[1]
    assert call_kwargs["model"] == "gemini-2.0-flash"


@pytest.mark.asyncio
async def test_run_agent_no_tools():
    """Test run_agent works with no tools (simple chat)."""
    mock_response = _make_response("STOP", [_make_text_part("Just chatting")])

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch("flyteplugins.gemini.agents._function_tools.genai.Client", return_value=mock_client):
        result = await run_agent(prompt="Hi", tools=[], api_key="test-key")

    assert result == "Just chatting"


@pytest.mark.asyncio
async def test_run_agent_tool_returns_non_string():
    """Test that non-string tool results are JSON serialized."""

    def get_data() -> dict:
        """Get data."""
        return {"temperature": 72, "unit": "F"}

    tool = function_tool(get_data)

    tool_response = _make_response(
        "STOP",
        [_make_function_call_part("get_data", {})],
    )
    final_response = _make_response("STOP", [_make_text_part("The temperature is 72F")])

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=[tool_response, final_response])

    with patch("flyteplugins.gemini.agents._function_tools.genai.Client", return_value=mock_client):
        result = await run_agent(prompt="temp?", tools=[tool], api_key="test-key")

    assert result == "The temperature is 72F"


@pytest.mark.asyncio
async def test_run_agent_multiple_tool_calls():
    """Test that multiple parallel function calls in one response are handled."""

    def tool_a(x: int) -> str:
        """Tool A."""
        return f"a:{x}"

    def tool_b(y: str) -> str:
        """Tool B."""
        return f"b:{y}"

    tools = [function_tool(tool_a), function_tool(tool_b)]

    # Gemini calls both functions in one response
    tool_response = _make_response(
        "STOP",
        [
            _make_function_call_part("tool_a", {"x": 1}),
            _make_function_call_part("tool_b", {"y": "hello"}),
        ],
    )
    final_response = _make_response("STOP", [_make_text_part("Both done")])

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=[tool_response, final_response])

    with patch("flyteplugins.gemini.agents._function_tools.genai.Client", return_value=mock_client):
        result = await run_agent(prompt="do both", tools=tools, api_key="test-key")

    assert result == "Both done"
    assert mock_client.aio.models.generate_content.call_count == 2
