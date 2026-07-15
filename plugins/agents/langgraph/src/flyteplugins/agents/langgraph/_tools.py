"""Turn Flyte tasks into LangGraph tools that execute as durable actions.

LangGraph accepts LangChain ``BaseTool`` instances as nodes/tools. :func:`tool`
wraps a Flyte ``@env.task`` as a LangChain ``StructuredTool`` whose ``arun``
dispatches to the task via ``task.aio()`` — so when the graph calls the tool,
it runs as a durable Flyte child action (its own container/resources, with
retries and caching) rather than inline in the graph's process.
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


@dataclasses.dataclass
class LangGraphTool:
    """A LangChain ``StructuredTool`` backed by a Flyte task."""

    structured_tool: typing.Any
    _task: AsyncFunctionTaskTemplate = dataclasses.field(repr=False)
    _name: str = ""

    @property
    def task(self) -> AsyncFunctionTaskTemplate:
        return self._task

    @property
    def __wrapped_task__(self) -> typing.Any:
        return self._task

    @property
    def __name__(self) -> str:
        return self._name

    async def __call__(self, **kwargs: typing.Any) -> str:
        """Execute the wrapped task as a durable child action."""
        result = await self._task.aio(**coerce_tool_args(self._task, kwargs or {}))
        return _as_content(result)


def tool(
    func: AsyncFunctionTaskTemplate | typing.Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> typing.Any:
    """Convert a Flyte task (or plain callable) into a LangChain ``StructuredTool``.

    - For an ``@env.task``: returns a ``StructuredTool`` whose ``arun`` runs the
      task as a durable Flyte child action when the graph invokes it. The input
      schema is derived from the task via the Flyte type engine. The backing task
      is wired to :class:`~flyteplugins.agents.core.ToolTaskResolver` and exposed
      via ``__wrapped_task__`` so it resolves to itself on the worker (no recursion).
    - For a plain (async) callable: returns a ``StructuredTool`` that runs it inline.

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
    """Build a LangChain ``StructuredTool`` from a Flyte task."""
    from langchain_core.tools import StructuredTool

    tool_name = name or task.func.__name__
    desc = (description or task.func.__doc__ or f"Run {tool_name}").strip()
    schema = task_json_schema(task)

    async def _arun(**kwargs: typing.Any) -> str:
        # In a Flyte task context this submits a durable child action; locally it
        # runs inline. ``coerce_tool_args`` relaxes LLM int->float args so Flyte's
        # type engine doesn't reject e.g. ``amount_usd=42`` for a ``float`` param.
        result = await task.aio(**coerce_tool_args(task, kwargs or {}))
        return _as_content(result)

    # Wire the shared resolver so the task resolves to itself on the worker.
    attach_tool_resolver(task)

    structured = StructuredTool.from_function(
        func=None,  # We use arun directly
        coroutine=_arun,
        name=tool_name,
        description=desc,
        args_schema=None,  # Use Flyte's schema
        schema=schema,
    )

    return LangGraphTool(structured_tool=structured, _task=task, _name=tool_name)


def _callable_to_tool(
    func: typing.Callable,
    *,
    name: str | None = None,
    description: str | None = None,
) -> typing.Any:
    """Build a LangChain ``StructuredTool`` from a plain callable."""
    from langchain_core.tools import StructuredTool

    tool_name = name or getattr(func, "__name__", "tool")
    desc = (description or func.__doc__ or f"Run {tool_name}").strip()
    schema = NativeInterface.from_callable(func).json_schema

    async def _arun(**kwargs: typing.Any) -> str:
        out = func(**(kwargs or {}))
        if inspect.isawaitable(out):
            out = await out
        return _as_content(out)

    return StructuredTool.from_function(
        func=None,
        coroutine=_arun,
        name=tool_name,
        description=desc,
        args_schema=None,
        schema=schema,
    )


def _as_content(result: typing.Any) -> str:
    """Convert a tool result to a string for LangChain's ToolMessage."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)


def _coerce_tool(t: typing.Any) -> typing.Any:
    """Coerce a tool to a LangChain-compatible tool."""
    if isinstance(t, AsyncFunctionTaskTemplate):
        return tool(t)
    return t
