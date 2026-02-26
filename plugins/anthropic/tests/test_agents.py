"""Unit tests for Anthropic agents plugin."""

import datetime
from typing import Literal, Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

import flyte
import pytest
from flyte._json_schema import literal_type_to_json_schema
from flyte.io import DataFrame, Dir, File
from flyte.models import NativeInterface
from flyte.types._type_engine import TypeEngine

from flyteplugins.anthropic import Agent, function_tool, run_agent
from flyteplugins.anthropic.agents._function_tools import FunctionTool

# ---------------------------------------------------------------------------
# literal_type_to_json_schema unit tests
# ---------------------------------------------------------------------------


def _schema(python_type) -> dict:
    """Helper: Python type → LiteralType → JSON schema."""
    return literal_type_to_json_schema(TypeEngine.to_literal_type(python_type))


def test_json_schema_string():
    assert _schema(str) == {"type": "string"}


def test_json_schema_int():
    assert _schema(int) == {"type": "integer"}


def test_json_schema_float():
    assert _schema(float) == {"type": "number", "format": "float"}


def test_json_schema_bool():
    assert _schema(bool) == {"type": "boolean"}


def test_json_schema_datetime():
    assert _schema(datetime.datetime) == {"type": "string", "format": "datetime"}


def test_json_schema_timedelta():
    assert _schema(datetime.timedelta) == {"type": "string", "format": "duration"}


def test_json_schema_list_str():
    assert _schema(list[str]) == {"type": "array", "items": {"type": "string"}}


def test_json_schema_bare_list():
    # Bare list without a type argument is not a standard Flyte type — the type
    # engine falls back to pickle. We just assert we get a non-empty schema dict
    # rather than raising an exception.
    schema = _schema(list)
    assert isinstance(schema, dict)
    assert len(schema) > 0


def test_json_schema_dict_str_int():
    assert _schema(dict[str, int]) == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }


def test_json_schema_bare_dict():
    schema = _schema(dict)
    assert schema["type"] == "object"


def test_json_schema_optional_str():
    # Optional[str] = Union[str, None] — simplified to just str schema
    assert _schema(Optional[str]) == {"type": "string"}


def test_json_schema_optional_list():
    # Optional[list[str]] — simplified to list schema
    assert _schema(Optional[list[str]]) == {"type": "array", "items": {"type": "string"}}


def test_json_schema_nested_list():
    assert _schema(list[list[int]]) == {
        "type": "array",
        "items": {"type": "array", "items": {"type": "integer"}},
    }


def test_json_schema_union_str_int():
    # True union — produces oneOf with format:union
    schema = _schema(Union[str, int])
    assert schema["format"] == "union"
    assert "oneOf" in schema
    types_in_schema = {v["type"] for v in schema["oneOf"]}
    assert types_in_schema == {"string", "integer"}


def test_json_schema_enum_from_literal():
    # Literal["a", "b"] → NativeInterface converts to Enum → enum_type LiteralType
    from flyte._interface import literal_to_enum

    enum_type = literal_to_enum(Literal["celsius", "fahrenheit"])
    schema = _schema(enum_type)
    assert schema == {"type": "string", "enum": ["celsius", "fahrenheit"]}


# ---------------------------------------------------------------------------
# File, Dir, DataFrame JSON schema tests
# ---------------------------------------------------------------------------


def test_json_schema_file():
    schema = _schema(File)
    assert schema["type"] == "object"
    assert schema["format"] == "blob"
    assert schema["properties"]["uri"] == {"type": "string", "default": ""}
    assert schema["properties"]["dimensionality"]["default"] == "SINGLE"


def test_json_schema_dir():
    schema = _schema(Dir)
    assert schema["type"] == "object"
    assert schema["format"] == "blob"
    assert schema["properties"]["uri"] == {"type": "string", "default": ""}
    assert schema["properties"]["dimensionality"]["default"] == "MULTIPART"


def test_json_schema_dataframe():
    schema = _schema(DataFrame)
    assert schema["type"] == "object"
    assert schema["format"] == "structured-dataset"
    assert "uri" in schema["properties"]
    assert "format" in schema["properties"]


def test_json_schema_optional_file():
    # Optional[File] should simplify to just the File schema
    schema = _schema(Optional[File])
    assert schema["type"] == "object"
    assert schema["format"] == "blob"


def test_native_interface_json_schema_with_file_and_dir():
    def process(input_file: File, output_dir: Dir) -> str:
        """Process a file and write to a directory."""
        return "done"

    schema = NativeInterface.from_callable(process).json_schema
    assert schema["properties"]["input_file"]["format"] == "blob"
    assert schema["properties"]["input_file"]["properties"]["dimensionality"]["default"] == "SINGLE"
    assert schema["properties"]["output_dir"]["format"] == "blob"
    assert schema["properties"]["output_dir"]["properties"]["dimensionality"]["default"] == "MULTIPART"
    assert set(schema["required"]) == {"input_file", "output_dir"}


def test_native_interface_json_schema_with_dataframe():
    def analyze(data: DataFrame, label: str) -> str:
        """Analyze a dataframe."""
        return label

    schema = NativeInterface.from_callable(analyze).json_schema
    assert schema["properties"]["data"]["format"] == "structured-dataset"
    assert schema["properties"]["label"] == {"type": "string"}
    assert set(schema["required"]) == {"data", "label"}


# ---------------------------------------------------------------------------
# NativeInterface.json_schema integration tests
# ---------------------------------------------------------------------------


def test_native_interface_json_schema_simple():
    def my_func(name: str, age: int) -> str:
        return f"{name} is {age}"

    schema = NativeInterface.from_callable(my_func).json_schema
    assert schema["type"] == "object"
    assert schema["properties"]["name"] == {"type": "string"}
    assert schema["properties"]["age"] == {"type": "integer"}
    assert set(schema["required"]) == {"name", "age"}


def test_native_interface_json_schema_with_default():
    def my_func(name: str, age: int = 25) -> str:
        return f"{name} is {age}"

    schema = NativeInterface.from_callable(my_func).json_schema
    assert schema["required"] == ["name"]
    assert "age" in schema["properties"]


def test_native_interface_json_schema_all_optional():
    def my_func(x: int = 0, y: int = 0) -> int:
        return x + y

    schema = NativeInterface.from_callable(my_func).json_schema
    assert schema["required"] == []


def test_native_interface_json_schema_literal_becomes_enum():
    def my_func(unit: Literal["C", "F"]) -> str:
        return unit

    schema = NativeInterface.from_callable(my_func).json_schema
    assert schema["properties"]["unit"] == {"type": "string", "enum": ["C", "F"]}
    assert "unit" in schema["required"]


def test_native_interface_json_schema_no_type_hints():
    def my_func(name, age):
        return f"{name} is {age}"

    schema = NativeInterface.from_callable(my_func).json_schema
    # Unannotated parameters fall back to string
    assert schema["properties"]["name"] == {"type": "string"}
    assert schema["properties"]["age"] == {"type": "string"}


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


def test_function_tool_flyte_task_uses_interface_schema():
    """Schema for a Flyte task comes from task.interface, not raw type hints."""
    env = flyte.TaskEnvironment("test-schema")

    @env.task
    def my_task(city: str, unit: Literal["C", "F"]) -> str:
        """Get weather."""
        return city

    tool = function_tool(my_task)
    schema = tool.input_schema

    assert schema["properties"]["city"] == {"type": "string"}
    # Literal["C","F"] must be an enum — this is the key correctness check
    assert schema["properties"]["unit"] == {"type": "string", "enum": ["C", "F"]}
    assert set(schema["required"]) == {"city", "unit"}


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
        model="claude-sonnet-4-20250514",
    )
    assert agent.name == "test_agent"
    assert agent.instructions == "You are a test agent."
    assert agent.max_tokens == 4096
    assert agent.max_iterations == 10


def test_agent_defaults():
    agent = Agent()
    assert agent.name == "assistant"
    assert agent.instructions == "You are a helpful assistant."
    assert agent.model == "claude-sonnet-4-20250514"
    assert agent.tools == []


def test_agent_with_tools():
    def my_tool(x: int) -> int:
        """Double the number."""
        return x * 2

    tool = function_tool(my_tool)
    agent = Agent(name="test", tools=[tool])

    anthropic_tools = agent.get_anthropic_tools()
    assert len(anthropic_tools) == 1
    assert anthropic_tools[0]["name"] == "my_tool"


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


@pytest.mark.asyncio
async def test_function_tool_execute_async_flyte_task_uses_aio():
    """Verify task.aio() is called (not self.func) for async Flyte tasks."""
    env = flyte.TaskEnvironment("test-exec-aio-path")

    @env.task
    async def add_async(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    tool = function_tool(add_async)

    with patch.object(tool.task, "aio", new_callable=AsyncMock, return_value=99) as mock_aio:
        result = await tool.execute(a=10, b=20)

    mock_aio.assert_called_once_with(a=10, b=20)
    assert result == 99


# ---------------------------------------------------------------------------
# Helpers for run_agent mock tests
# ---------------------------------------------------------------------------


def _make_text_block(text: str):
    """Create a mock text content block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(tool_id: str, name: str, tool_input: dict):
    """Create a mock tool_use content block."""
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = tool_input
    return block


def _make_response(stop_reason: str, content: list):
    """Create a mock Claude API response."""
    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = content
    return response


# ---------------------------------------------------------------------------
# run_agent tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_agent_missing_api_key():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="Anthropic API key is required"):
            await run_agent(prompt="hello", api_key=None)


@pytest.mark.asyncio
async def test_run_agent_simple_text_response():
    mock_response = _make_response("end_turn", [_make_text_block("Hello!")])

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("flyteplugins.anthropic.agents._function_tools.anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_agent(prompt="Hi", api_key="test-key")

    assert result == "Hello!"


@pytest.mark.asyncio
async def test_run_agent_empty_text_response():
    mock_response = _make_response("end_turn", [])

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("flyteplugins.anthropic.agents._function_tools.anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_agent(prompt="Hi", api_key="test-key")

    assert result == ""


@pytest.mark.asyncio
async def test_run_agent_with_tool_call():
    """Test the full tool call loop: tool_use -> end_turn."""

    def get_weather(city: str) -> str:
        """Get weather."""
        return f"Sunny in {city}"

    tool = function_tool(get_weather)

    # First response: Claude wants to use a tool
    tool_response = _make_response(
        "tool_use",
        [
            _make_text_block("Let me check the weather."),
            _make_tool_use_block("call_1", "get_weather", {"city": "SF"}),
        ],
    )

    # Second response: Claude gives final answer
    final_response = _make_response("end_turn", [_make_text_block("It's sunny in SF!")])

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(side_effect=[tool_response, final_response])

    with patch("flyteplugins.anthropic.agents._function_tools.anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_agent(prompt="Weather in SF?", tools=[tool], api_key="test-key")

    assert result == "It's sunny in SF!"
    assert mock_client.messages.create.call_count == 2


@pytest.mark.asyncio
async def test_run_agent_unknown_tool():
    """Test that unknown tool names are handled gracefully with is_error."""

    tool_response = _make_response(
        "tool_use",
        [_make_tool_use_block("call_1", "nonexistent_tool", {})],
    )
    final_response = _make_response("end_turn", [_make_text_block("Sorry, tool not found.")])

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(side_effect=[tool_response, final_response])

    with patch("flyteplugins.anthropic.agents._function_tools.anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_agent(prompt="test", tools=[], api_key="test-key")

    assert result == "Sorry, tool not found."

    # Verify the tool_result was sent with is_error
    second_call_messages = mock_client.messages.create.call_args_list[1][1]["messages"]
    tool_result_msg = second_call_messages[-1]  # last message should be the tool result
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"][0]["is_error"] is True


@pytest.mark.asyncio
async def test_run_agent_tool_execution_error():
    """Test that tool execution errors are caught and sent back to Claude."""

    def failing_tool(x: int) -> str:
        """Always fails."""
        raise RuntimeError("Something broke")

    tool = function_tool(failing_tool)

    tool_response = _make_response(
        "tool_use",
        [_make_tool_use_block("call_1", "failing_tool", {"x": 1})],
    )
    final_response = _make_response("end_turn", [_make_text_block("Tool failed, sorry.")])

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(side_effect=[tool_response, final_response])

    with patch("flyteplugins.anthropic.agents._function_tools.anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_agent(prompt="test", tools=[tool], api_key="test-key")

    assert result == "Tool failed, sorry."


@pytest.mark.asyncio
async def test_run_agent_max_iterations():
    """Test that the agent stops after max_iterations."""

    def dummy(x: int) -> str:
        """Dummy."""
        return str(x)

    tool = function_tool(dummy)

    # Always returns tool_use, never ends
    infinite_tool_response = _make_response(
        "tool_use",
        [_make_tool_use_block("call_1", "dummy", {"x": 1})],
    )

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=infinite_tool_response)

    with patch("flyteplugins.anthropic.agents._function_tools.anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_agent(prompt="test", tools=[tool], api_key="test-key", max_iterations=3)

    assert result == "Maximum iterations reached without final response."
    assert mock_client.messages.create.call_count == 3


@pytest.mark.asyncio
async def test_run_agent_max_tokens_stop_reason():
    """Test that max_tokens stop_reason returns partial text."""
    mock_response = _make_response("max_tokens", [_make_text_block("Partial response that was cut")])

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("flyteplugins.anthropic.agents._function_tools.anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_agent(prompt="test", api_key="test-key")

    assert result == "Partial response that was cut"


@pytest.mark.asyncio
async def test_run_agent_stop_sequence_reason():
    """Test that stop_sequence stop_reason returns text."""
    mock_response = _make_response("stop_sequence", [_make_text_block("Stopped here")])

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("flyteplugins.anthropic.agents._function_tools.anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_agent(prompt="test", api_key="test-key")

    assert result == "Stopped here"


@pytest.mark.asyncio
async def test_run_agent_unexpected_stop_no_text():
    """Test that unexpected stop_reason with no text returns informative message."""
    mock_response = _make_response("refusal", [])

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("flyteplugins.anthropic.agents._function_tools.anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_agent(prompt="test", api_key="test-key")

    assert "refusal" in result


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
        model="claude-haiku-3-20240307",
        tools=[tool],
        max_tokens=1024,
        max_iterations=5,
    )

    mock_response = _make_response("end_turn", [_make_text_block("Done")])

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("flyteplugins.anthropic.agents._function_tools.anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_agent(prompt="test", agent=agent, api_key="test-key")

    assert result == "Done"
    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-haiku-3-20240307"
    assert call_kwargs["max_tokens"] == 1024
    assert call_kwargs["system"] == "Be precise."


@pytest.mark.asyncio
async def test_run_agent_no_tools():
    """Test run_agent works with no tools (simple chat)."""
    mock_response = _make_response("end_turn", [_make_text_block("Just chatting")])

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("flyteplugins.anthropic.agents._function_tools.anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_agent(prompt="Hi", tools=[], api_key="test-key")

    assert result == "Just chatting"
    # Verify tools param was NOT included in the API call
    call_kwargs = mock_client.messages.create.call_args[1]
    assert "tools" not in call_kwargs


@pytest.mark.asyncio
async def test_run_agent_tool_returns_non_string():
    """Test that non-string tool results are JSON serialized."""

    def get_data() -> dict:
        """Get data."""
        return {"temperature": 72, "unit": "F"}

    tool = function_tool(get_data)

    tool_response = _make_response(
        "tool_use",
        [_make_tool_use_block("call_1", "get_data", {})],
    )
    final_response = _make_response("end_turn", [_make_text_block("The temperature is 72F")])

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(side_effect=[tool_response, final_response])

    with patch("flyteplugins.anthropic.agents._function_tools.anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_agent(prompt="temp?", tools=[tool], api_key="test-key")

    assert result == "The temperature is 72F"

    # Verify the tool result was JSON serialized
    second_call_messages = mock_client.messages.create.call_args_list[1][1]["messages"]
    tool_result_content = second_call_messages[-1]["content"][0]["content"]
    assert '"temperature"' in tool_result_content
    assert '"unit"' in tool_result_content


@pytest.mark.asyncio
async def test_run_agent_multiple_tool_calls():
    """Test that multiple parallel tool calls in one response are handled."""

    def tool_a(x: int) -> str:
        """Tool A."""
        return f"a:{x}"

    def tool_b(y: str) -> str:
        """Tool B."""
        return f"b:{y}"

    tools = [function_tool(tool_a), function_tool(tool_b)]

    # Claude calls both tools in one response
    tool_response = _make_response(
        "tool_use",
        [
            _make_tool_use_block("call_1", "tool_a", {"x": 1}),
            _make_tool_use_block("call_2", "tool_b", {"y": "hello"}),
        ],
    )
    final_response = _make_response("end_turn", [_make_text_block("Both done")])

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(side_effect=[tool_response, final_response])

    with patch("flyteplugins.anthropic.agents._function_tools.anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_agent(prompt="do both", tools=tools, api_key="test-key")

    assert result == "Both done"

    # Verify both tool results were sent
    second_call_messages = mock_client.messages.create.call_args_list[1][1]["messages"]
    tool_results = second_call_messages[-1]["content"]
    assert len(tool_results) == 2
    assert tool_results[0]["tool_use_id"] == "call_1"
    assert tool_results[1]["tool_use_id"] == "call_2"
