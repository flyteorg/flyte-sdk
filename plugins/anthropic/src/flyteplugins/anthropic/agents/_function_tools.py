"""Anthropic Claude tool integration for Flyte tasks.

This module provides utilities to convert Flyte tasks into Anthropic tool definitions
and run Claude agents with those tools.
"""

import inspect
import json
import os
import typing
from dataclasses import dataclass, field
from functools import partial

import anthropic
from flyte._task import AsyncFunctionTaskTemplate, TaskTemplate

# Type mapping from Python types to JSON schema types
TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(py_type: type) -> dict[str, typing.Any]:
    """Convert a Python type hint to a JSON schema definition."""
    origin = typing.get_origin(py_type)
    args = typing.get_args(py_type)

    # Handle Optional types
    if origin is typing.Union:
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            schema = _python_type_to_json_schema(non_none_args[0])
            return schema
        return {"anyOf": [_python_type_to_json_schema(a) for a in non_none_args]}

    # Handle list[X]
    if origin is list:
        if args:
            return {"type": "array", "items": _python_type_to_json_schema(args[0])}
        return {"type": "array"}

    # Handle dict[K, V]
    if origin is dict:
        return {"type": "object"}

    # Handle basic types
    if py_type in TYPE_MAP:
        return {"type": TYPE_MAP[py_type]}

    # Default to string for unknown types
    return {"type": "string"}


def _get_function_schema(func: typing.Callable) -> dict[str, typing.Any]:
    """Extract JSON schema from a function's type hints and docstring."""
    sig = inspect.signature(func)
    hints = typing.get_type_hints(func)

    properties: dict[str, typing.Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        param_type = hints.get(name, str)
        prop_schema = _python_type_to_json_schema(param_type)

        # Add description from docstring if available
        properties[name] = prop_schema

        # Check if parameter is required (no default value)
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


@dataclass
class FunctionTool:
    """A Flyte-compatible tool definition for Anthropic Claude.

    This dataclass represents a tool that can be used with Claude's tool use API.
    It wraps a Flyte task and provides the necessary schema for Claude to invoke it.
    """

    name: str
    description: str
    input_schema: dict[str, typing.Any]
    func: typing.Callable
    task: TaskTemplate | None = None
    is_async: bool = False

    def to_anthropic_tool(self) -> dict[str, typing.Any]:
        """Convert to Anthropic tool definition format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    async def execute(self, **kwargs) -> typing.Any:
        """Execute the tool with the given arguments."""
        if self.task is not None:
            if self.is_async:
                return await self.task(**kwargs)
            return self.task(**kwargs)
        if self.is_async:
            return await self.func(**kwargs)
        return self.func(**kwargs)


def function_tool(
    func: AsyncFunctionTaskTemplate | typing.Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> FunctionTool:
    """Convert a function or Flyte task to an Anthropic-compatible tool.

    This decorator/function converts a Python function or Flyte task into a
    FunctionTool that can be used with Claude's tool use API.

    Args:
        func: The function or Flyte task to convert.
        name: Optional custom name for the tool. Defaults to the function name.
        description: Optional custom description. Defaults to the function's docstring.

    Returns:
        A FunctionTool instance that can be used with run_agent().

    Example:
        ```python
        @env.task
        async def get_weather(city: str) -> str:
            '''Get the current weather for a city.'''
            return f"Weather in {city}: sunny"

        tool = function_tool(get_weather)
        ```
    """
    if func is None:
        return partial(function_tool, name=name, description=description)

    # Handle Flyte tasks
    if isinstance(func, AsyncFunctionTaskTemplate):
        actual_func = func.func
        task = func
    else:
        actual_func = func
        task = None

    tool_name = name or actual_func.__name__
    tool_description = description or (actual_func.__doc__ or f"Execute {tool_name}")
    input_schema = _get_function_schema(actual_func)
    is_async = inspect.iscoroutinefunction(actual_func)

    return FunctionTool(
        name=tool_name,
        description=tool_description.strip(),
        input_schema=input_schema,
        func=actual_func,
        task=task,
        is_async=is_async,
    )


@dataclass
class Agent:
    """A Claude agent configuration.

    This class represents the configuration for a Claude agent, including
    the model to use, system instructions, and available tools.
    """

    name: str = "assistant"
    instructions: str = "You are a helpful assistant."
    model: str = "claude-sonnet-4-20250514"
    tools: list[FunctionTool] = field(default_factory=list)
    max_tokens: int = 4096
    max_iterations: int = 10

    def get_anthropic_tools(self) -> list[dict[str, typing.Any]]:
        """Get tool definitions in Anthropic format."""
        return [tool.to_anthropic_tool() for tool in self.tools]


async def run_agent(
    prompt: str,
    tools: list[FunctionTool] | None = None,
    *,
    agent: Agent | None = None,
    model: str = "claude-sonnet-4-20250514",
    system: str | None = None,
    max_tokens: int = 4096,
    max_iterations: int = 10,
    api_key: str | None = None,
) -> str:
    """Run a Claude agent with the given tools and prompt.

    This function creates a Claude conversation loop that can use tools
    to accomplish tasks. It handles the back-and-forth of tool calls
    and responses until the agent produces a final text response.

    Args:
        prompt: The user prompt to send to the agent.
        tools: List of FunctionTool instances to make available to the agent.
        agent: Optional Agent configuration. If provided, overrides other params.
        model: The Claude model to use.
        system: Optional system prompt.
        max_tokens: Maximum tokens in the response.
        max_iterations: Maximum number of tool call iterations.
        api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.

    Returns:
        The final text response from the agent.

    Example:
        ```python
        result = await run_agent(
            prompt="What's the weather in SF?",
            tools=[function_tool(get_weather)],
        )
        ```
    """
    if agent is not None:
        tools = agent.tools
        model = agent.model
        system = agent.instructions
        max_tokens = agent.max_tokens
        max_iterations = agent.max_iterations

    tools = tools or []
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError(
            "Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable "
            "or pass api_key parameter."
        )

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Build tool definitions
    anthropic_tools = [tool.to_anthropic_tool() for tool in tools]
    tool_map = {tool.name: tool for tool in tools}

    # Initialize conversation
    messages: list[dict[str, typing.Any]] = [{"role": "user", "content": prompt}]

    for _ in range(max_iterations):
        # Call Claude
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or "You are a helpful assistant.",
            tools=anthropic_tools if anthropic_tools else anthropic.NOT_GIVEN,
            messages=messages,
        )

        # Check if we're done (no tool use)
        if response.stop_reason == "end_turn":
            # Extract final text response
            for block in response.content:
                if block.type == "text":
                    return block.text
            return ""

        # Process tool calls
        tool_results = []
        assistant_content = []

        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

                # Execute the tool
                tool = tool_map.get(block.name)
                if tool is None:
                    result = f"Error: Unknown tool '{block.name}'"
                else:
                    try:
                        result = await tool.execute(**block.input)
                        if not isinstance(result, str):
                            result = json.dumps(result)
                    except Exception as e:
                        result = f"Error executing tool: {e}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        # Add assistant message and tool results to conversation
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": tool_results})

    # Max iterations reached
    return "Maximum iterations reached without final response."
