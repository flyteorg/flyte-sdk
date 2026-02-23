"""Anthropic Claude tool integration for Flyte tasks.

This module provides utilities to convert Flyte tasks into Anthropic tool definitions
and run Claude agents with those tools.
"""

import asyncio
import inspect
import json
import logging
import os
import typing
from dataclasses import dataclass, field
from functools import partial

from flyte._task import AsyncFunctionTaskTemplate
from flyte.models import NativeInterface

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class FunctionTool:
    """A Flyte-compatible tool definition for Anthropic Claude.

    This dataclass represents a tool that can be used with Claude's tool use API.
    It wraps a Flyte task or regular callable and provides the necessary schema
    for Claude to invoke it.
    """

    name: str
    description: str
    input_schema: dict[str, typing.Any]
    func: typing.Callable
    task: AsyncFunctionTaskTemplate | None = None
    native_interface: NativeInterface | None = None
    report: bool = False
    is_async: bool = False

    def to_anthropic_tool(self) -> dict[str, typing.Any]:
        """Convert to Anthropic tool definition format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    async def execute(self, **kwargs) -> typing.Any:
        """Execute the tool with the given arguments.

        Async functions are awaited directly. Sync functions are run in a
        thread executor to avoid blocking the event loop.
        """
        if self.task is not None:
            if self.is_async:
                return await self.task(**kwargs)
            return await asyncio.to_thread(self.task, **kwargs)
        if self.is_async:
            return await self.func(**kwargs)
        return await asyncio.to_thread(self.func, **kwargs)


def function_tool(
    func: AsyncFunctionTaskTemplate | typing.Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> "FunctionTool | partial[FunctionTool]":
    """Convert a function or Flyte task to an Anthropic-compatible tool.

    This function converts a Python function, @flyte.trace decorated function,
    or Flyte task into a FunctionTool that can be used with Claude's tool use API.

    The input_schema is derived via the Flyte type engine, producing JSON schema
    This ensures that Literal types, dataclasses, FlyteFile, and other Flyte-native
    types are represented correctly.

    For @flyte.trace decorated functions, the tracing context is preserved
    automatically since functools.wraps maintains the original function's metadata.

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
        native_interface = func.interface
        report = func.report
        # Use the task's already-built NativeInterface — goes through the type
        # engine so Literal enums, dataclasses, FlyteFile etc. are all correct.
        input_schema = func.json_schema
    else:
        # Regular callables and @flyte.trace decorated functions.
        # @flyte.trace uses functools.wraps, so __name__, __doc__ and type hints
        # are preserved. The tracing activates automatically in a task context.
        actual_func = func
        task = None
        native_interface = None
        report = False
        # Build a NativeInterface on the fly so the same type-engine path is used.
        input_schema = NativeInterface.from_callable(actual_func).json_schema

    tool_name = name or actual_func.__name__
    tool_description = description or (actual_func.__doc__ or f"Execute {tool_name}")
    is_async = inspect.iscoroutinefunction(actual_func)

    return FunctionTool(
        name=tool_name,
        description=tool_description.strip(),
        input_schema=input_schema,
        func=actual_func,
        task=task,
        native_interface=native_interface,
        report=report,
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
            "Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
        )

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Build tool definitions
    anthropic_tools = [tool.to_anthropic_tool() for tool in tools]
    tool_map = {tool.name: tool for tool in tools}

    # Initialize conversation
    messages: list[dict[str, typing.Any]] = [{"role": "user", "content": prompt}]

    # Build base kwargs for the API call
    create_kwargs: dict[str, typing.Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system or "You are a helpful assistant.",
        "messages": messages,
    }
    if anthropic_tools:
        create_kwargs["tools"] = anthropic_tools

    for _ in range(max_iterations):
        # Call Claude
        create_kwargs["messages"] = messages
        response = await client.messages.create(**create_kwargs)

        # Extract text from response content
        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    return block.text
            return ""

        # Handle non-tool-use stop reasons (max_tokens, stop_sequence, refusal)
        if response.stop_reason != "tool_use":
            text_parts = [block.text for block in response.content if block.type == "text"]
            if text_parts:
                return " ".join(text_parts)
            return f"Agent stopped unexpectedly: {response.stop_reason}"

        # Process tool calls (stop_reason == "tool_use")
        tool_results = []
        assistant_content = []

        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

                # Execute the tool
                tool = tool_map.get(block.name)
                if tool is None:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Error: Unknown tool '{block.name}'",
                            "is_error": True,
                        }
                    )
                else:
                    try:
                        result = await tool.execute(**block.input)
                        if not isinstance(result, str):
                            result = json.dumps(result)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            }
                        )
                    except Exception:
                        logger.exception("Error executing tool '%s'", block.name)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"Error executing tool '{block.name}'",
                                "is_error": True,
                            }
                        )

        # Add assistant message and tool results to conversation
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": tool_results})

    # Max iterations reached
    return "Maximum iterations reached without final response."
