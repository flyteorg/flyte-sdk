"""Turn Flyte tasks into CrewAI tools that execute as durable actions.

CrewAI accepts tool callables as tools. :func:`tool` wraps a Flyte ``@env.task``
as a CrewAI tool whose execution dispatches to the task via ``task.aio()`` — so
when the agent calls the tool, it runs as a durable Flyte child action (its own
container/resources, with retries and caching) rather than inline in the agent's
process.
"""

from __future__ import annotations

import inspect
import json
import typing
from functools import partial

from flyte._task import AsyncFunctionTaskTemplate
from flyte.models import NativeInterface
from flyteplugins.agents.core import attach_tool_resolver, coerce_tool_args, task_json_schema


def tool(
    func: AsyncFunctionTaskTemplate | typing.Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> typing.Any:
    """Convert a Flyte task (or plain callable) into a CrewAI tool.

    - For an ``@env.task``: returns a CrewAI tool whose execution runs the task as
      a durable Flyte child action when the agent invokes it. The input schema is
      derived from the task via the Flyte type engine. The backing task is wired
      to :class:`~flyteplugins.agents.core.ToolTaskResolver` and exposed via
      ``__wrapped_task__`` so it resolves to itself on the worker (no recursion).
    - For a plain (async) callable: returns a CrewAI tool that runs it inline.

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
    """Build a CrewAI tool from a Flyte task."""
    tool_name = name or task.func.__name__
    desc = (description or task.func.__doc__ or f"Run {tool_name}").strip()
    task_json_schema(task)  # validate schema at construction time

    async def _execute(**kwargs: typing.Any) -> str:
        # In a Flyte task context this submits a durable child action; locally it
        # runs inline. ``coerce_tool_args`` relaxes LLM int->float args so Flyte's
        # type engine doesn't reject e.g. ``amount_usd=42`` for a ``float`` param.
        result = await task.aio(**coerce_tool_args(task, kwargs or {}))
        return _as_content(result)

    # Wire the shared resolver so the task resolves to itself on the worker.
    attach_tool_resolver(task)

    # CrewAI tools are typically plain async callables with a name/description.
    # We set the __name__ and __doc__ attributes for schema derivation.
    _execute.__name__ = tool_name  # type: ignore[attr-defined]
    _execute.__doc__ = desc  # type: ignore[attr-defined]
    _execute.__wrapped_task__ = task  # type: ignore[attr-defined]
    _execute.task = task  # type: ignore[attr-defined]
    return _execute


def _callable_to_tool(
    func: typing.Callable,
    *,
    name: str | None = None,
    description: str | None = None,
) -> typing.Any:
    """Build a CrewAI tool from a plain callable."""
    tool_name = name or getattr(func, "__name__", "tool")
    desc = (description or func.__doc__ or f"Run {tool_name}").strip()
    NativeInterface.from_callable(func).json_schema  # validate schema at construction time

    async def _execute(**kwargs: typing.Any) -> str:
        out = func(**(kwargs or {}))
        if inspect.isawaitable(out):
            out = await out
        return _as_content(out)

    _execute.__name__ = tool_name  # type: ignore[attr-defined]
    _execute.__doc__ = desc  # type: ignore[attr-defined]
    return _execute


def _as_content(result: typing.Any) -> str:
    """Convert a tool result to a string for CrewAI."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)


def _coerce_tool(t: typing.Any) -> typing.Any:
    """Coerce a tool to a CrewAI-compatible tool."""
    if isinstance(t, AsyncFunctionTaskTemplate):
        return tool(t)
    return t
