"""Turn Flyte tasks into Pydantic AI tools that execute as durable actions.

Pydantic AI accepts ``Tool`` instances (or plain callables) as tools. :func:`tool`
wraps a Flyte ``@env.task`` as a Pydantic AI ``Tool`` whose execution dispatches
to the task via ``task.aio()`` — so when the agent calls the tool, it runs as a
durable Flyte child action (its own container/resources, with retries and caching)
rather than inline in the agent's process.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import typing
from functools import partial

from flyte._task import AsyncFunctionTaskTemplate
from flyte.models import NativeInterface
from flyteplugins.agents.core import attach_tool_resolver, coerce_tool_args, task_json_schema

from pydantic_ai import Tool
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models import Model


@dataclasses.dataclass
class FunctionTool:
    """A Pydantic AI ``Tool`` backed by a Flyte task."""

    task: AsyncFunctionTaskTemplate
    tool_name: str
    desc: str

    async def execute(self, call: ToolCallPart, model: Model) -> ToolReturnPart:
        args_dict = call.args_as_dict() if hasattr(call, "args_as_dict") else call.args
        result = await self.task.aio(**coerce_tool_args(self.task, dict(args_dict) if args_dict else {}))
        content = _as_content(result)
        return ToolReturnPart(tool_name=call.tool_name, content=content, result_id=call.id)

    @property
    def __wrapped_task__(self) -> typing.Any:
        return self.task


def tool(
    func: AsyncFunctionTaskTemplate | typing.Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> typing.Any:
    """Convert a Flyte task (or plain callable) into a Pydantic AI ``Tool``.

    - For an ``@env.task``: returns a ``Tool`` whose execution runs the task as a
      durable Flyte child action when the agent invokes it. The input schema is
      derived from the task via the Flyte type engine. The backing task is wired
      to :class:`~flyteplugins.agents.core.ToolTaskResolver` and exposed via
      ``__wrapped_task__`` so it resolves to itself on the worker (no recursion).
    - For a plain (async) callable: returns a ``Tool`` that runs it inline.

    Usable bare, parametrized, or as a direct call::

        @tool
        @env.task
        async def get_weather(city: str) -> str: ...
    """
    if func is None:
        return partial(tool, name=name, description=description)
    if isinstance(func, AsyncFunctionTaskTemplate):
        return _task_to_tool(func, name=name, description=description)
    return _callable_to_tool(func, name=name, description=description)


def _task_to_tool(
    task: AsyncFunctionTaskTemplate,
    *,
    name: str | None = None,
    description: str | None = None,
) -> typing.Any:
    """Build a Pydantic AI ``Tool`` from a Flyte task.

    Returns the original function with extra attributes so it:
    - passes ``inspect.isfunction()``
    - is directly callable (dispatches to ``task.aio()``)
    - exposes ``__wrapped_task__`` and ``.task`` for conformance checks
    """
    tool_name = name or task.func.__name__
    _ = (description or task.func.__doc__ or f"Run {tool_name}").strip()
    task_json_schema(task)  # validate schema at construction time

    # Wire the shared resolver so the task resolves to itself on the worker.
    attach_tool_resolver(task)

    func = task.func

    async def _wrapper(**kwargs: typing.Any) -> typing.Any:
        return await task.aio(**kwargs)

    _wrapper.__name__ = tool_name
    _wrapper.__qualname__ = func.__qualname__
    _wrapper.__doc__ = func.__doc__
    _wrapper.__wrapped_task__ = task
    _wrapper.task = task

    return typing.cast(typing.Any, _wrapper)


def _callable_to_tool(
    func: typing.Callable,
    *,
    name: str | None = None,
    description: str | None = None,
) -> typing.Any:
    """Build a Pydantic AI ``Tool`` from a plain callable."""
    tool_name = name or getattr(func, "__name__", "tool")
    desc = (description or func.__doc__ or f"Run {tool_name}").strip()
    NativeInterface.from_callable(func).json_schema  # validate schema at construction time

    async def _execute(call: ToolCallPart, model: Model) -> ToolReturnPart:
        args_dict = call.args_as_dict() if hasattr(call, "args_as_dict") else call.args
        out = func(**(dict(args_dict) if args_dict else {}))
        if inspect.isawaitable(out):
            out = await out
        content = _as_content(out)
        return ToolReturnPart(tool_name=call.tool_name, content=content, result_id=call.id)

    return Tool(
        tool_name,
        desc,
        _execute,
        deps_type=None,
        result_type=str,
        return_direct=False,
    )


def _as_content(result: typing.Any) -> str:
    """Convert a tool result to a string for Pydantic AI's ToolReturnPart."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)


def _coerce_tool(t: typing.Any) -> typing.Any:
    """Coerce a tool to a Pydantic AI-compatible tool."""
    if isinstance(t, AsyncFunctionTaskTemplate):
        return tool(t)
    return t
