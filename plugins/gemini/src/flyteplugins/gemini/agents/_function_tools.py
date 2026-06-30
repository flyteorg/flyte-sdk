"""Google Gemini tool integration for Flyte tasks.

This module provides utilities to convert Flyte tasks into Gemini tool definitions
and run Gemini agents with those tools.
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
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


@dataclass
class FunctionTool:
    """A Flyte-compatible tool definition for Google Gemini.

    This dataclass represents a tool that can be used with Gemini's function calling API.
    It wraps a Flyte task or regular callable and provides the necessary schema
    for Gemini to invoke it.
    """

    name: str
    description: str
    input_schema: dict[str, typing.Any]
    func: typing.Callable
    task: AsyncFunctionTaskTemplate | None = None
    native_interface: NativeInterface | None = None
    report: bool = False
    is_async: bool = False

    def to_gemini_tool(self) -> types.FunctionDeclaration:
        """Convert to Gemini FunctionDeclaration format."""
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=self.input_schema,
        )

    async def execute(self, **kwargs) -> typing.Any:
        """Execute the tool with the given arguments.

        Async functions are awaited directly. Sync functions are run in a
        thread executor to avoid blocking the event loop.
        """
        if self.task is not None:
            if self.is_async:
                return await self.task.aio(**kwargs)
        if self.is_async:
            return await self.func(**kwargs)
        return await asyncio.to_thread(self.func, **kwargs)


def function_tool(
    func: AsyncFunctionTaskTemplate | typing.Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> "FunctionTool | partial[FunctionTool]":
    """Convert a function or Flyte task to a Gemini-compatible tool.

    This function converts a Python function, @flyte.trace decorated function,
    or Flyte task into a FunctionTool that can be used with Gemini's function calling API.

    The input_schema is derived via the Flyte type engine, producing JSON schema.
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
    """A Gemini agent configuration.

    This class represents the configuration for a Gemini agent, including
    the model to use, system instructions, and available tools.

    Attributes:
        name: A human-readable name for this agent. Used for logging and
            identification only; not sent to the API.
        instructions: The system prompt passed to Gemini on every turn.
            Describes the agent's role, tone, and constraints.
        model: The Gemini model ID to use, e.g. `"gemini-2.5-flash"`.
        tools: List of `FunctionTool` instances the agent can invoke.
            Create tools with `function_tool()`.
        max_output_tokens: Maximum number of tokens in each Gemini response.
        max_iterations: Maximum number of function-call / response cycles before
            `run_agent` returns with a timeout message.
    """

    name: str = "assistant"
    instructions: str = "You are a helpful assistant."
    model: str = "gemini-2.5-flash"
    tools: list[FunctionTool] = field(default_factory=list)
    max_output_tokens: int = 8192
    max_iterations: int = 10

    def get_gemini_tools(self) -> list[types.FunctionDeclaration]:
        """Get tool definitions in Gemini format."""
        return [tool.to_gemini_tool() for tool in self.tools]


async def run_agent(
    prompt: str,
    tools: list[FunctionTool] | None = None,
    *,
    agent: Agent | None = None,
    model: str = "gemini-2.5-flash",
    system: str | None = None,
    max_output_tokens: int = 8192,
    max_iterations: int = 10,
    api_key: str | None = None,
) -> str:
    """Run a Gemini agent with the given tools and prompt.

    This function creates a Gemini conversation loop that can use tools
    to accomplish tasks. It handles the back-and-forth of function calls
    and responses until the agent produces a final text response.

    Args:
        prompt: The user prompt to send to the agent.
        tools: List of FunctionTool instances to make available to the agent.
        agent: Optional Agent configuration. If provided, overrides other params.
        model: The Gemini model to use.
        system: Optional system prompt.
        max_output_tokens: Maximum tokens in the response.
        max_iterations: Maximum number of tool call iterations.
        api_key: Google API key. Defaults to GOOGLE_API_KEY env var.

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
        max_output_tokens = agent.max_output_tokens
        max_iterations = agent.max_iterations

    tools = tools or []
    api_key = api_key or os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter."
        )

    client = genai.Client(api_key=api_key)

    # Build tool definitions
    gemini_tools = [tool.to_gemini_tool() for tool in tools]
    tool_map = {tool.name: tool for tool in tools}

    # Build config
    config_kwargs: dict[str, typing.Any] = {
        "max_output_tokens": max_output_tokens,
        "system_instruction": system or "You are a helpful assistant.",
    }
    if gemini_tools:
        config_kwargs["tools"] = [types.Tool(function_declarations=gemini_tools)]
        config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)

    config = types.GenerateContentConfig(**config_kwargs)

    # Initialize conversation
    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)]),
    ]

    for _ in range(max_iterations):
        # Call Gemini
        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason

        # Check for function calls in the response
        function_calls = []
        text_parts = []
        for part in candidate.content.parts:
            if part.function_call is not None:
                function_calls.append(part)
            elif part.text is not None:
                text_parts.append(part.text)

        # No function calls - return text response
        if not function_calls:
            if text_parts:
                return " ".join(text_parts)
            if finish_reason == "SAFETY":
                return "Agent stopped: content was blocked by safety filters."
            if finish_reason == "MAX_TOKENS":
                return " ".join(text_parts) if text_parts else "Agent stopped: maximum output tokens reached."
            return f"Agent stopped unexpectedly: {finish_reason}"

        # Add model response to conversation
        contents.append(candidate.content)

        # Process function calls and build responses
        function_response_parts = []
        for part in function_calls:
            fc = part.function_call
            tool = tool_map.get(fc.name)

            if tool is None:
                function_response_parts.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"error": f"Unknown tool '{fc.name}'"},
                    )
                )
            else:
                try:
                    args = dict(fc.args) if fc.args else {}
                    result = await tool.execute(**args)
                    if isinstance(result, str):
                        response_data = {"result": result}
                    else:
                        response_data = {"result": json.dumps(result)}
                    function_response_parts.append(
                        types.Part.from_function_response(
                            name=fc.name,
                            response=response_data,
                        )
                    )
                except Exception:
                    logger.exception("Error executing tool '%s'", fc.name)
                    function_response_parts.append(
                        types.Part.from_function_response(
                            name=fc.name,
                            response={"error": f"Error executing tool '{fc.name}'"},
                        )
                    )

        # Add function responses to conversation
        contents.append(types.Content(role="user", parts=function_response_parts))

    # Max iterations reached
    return "Maximum iterations reached without final response."
