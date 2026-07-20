"""Turn Flyte tasks into LangGraph-compatible tools that run as durable actions.

LangGraph (via LangChain) drives tools that are ``BaseTool`` instances: it binds
them to the model (``model.bind_tools([...])``) so the LLM can call them, and it
executes them from a tool node. :func:`tool` wraps a Flyte ``@env.task`` as a
LangChain ``StructuredTool`` whose async body dispatches to the task via
``task.aio()`` — so when the graph executes the tool, it runs as a durable Flyte
child action (its own container/resources, with retries and caching) rather than
inline in the graph's process.

The returned tool is a first-class ``StructuredTool``: pass it straight to
``model.bind_tools(...)``, to :func:`~flyteplugins.agents.langgraph.tool_node`,
or to LangGraph's ``ToolNode``. It additionally exposes ``__wrapped_task__`` /
``task`` (so the backing task resolves to itself on the worker, no recursion) and
``__name__`` for convenience.
"""

from __future__ import annotations

import functools
import inspect
import json
import typing
from functools import partial

from flyte._task import AsyncFunctionTaskTemplate
from flyteplugins.agents.core import attach_tool_resolver, coerce_tool_args

try:  # pragma: no cover - import shape only
    from langchain_core.tools import StructuredTool
except Exception:  # pragma: no cover
    StructuredTool = None  # type: ignore[assignment,misc]


if StructuredTool is not None:

    class FlyteStructuredTool(StructuredTool):
        """A LangChain ``StructuredTool`` backed by a Flyte task.

        Behaves exactly like a ``StructuredTool`` (so LangGraph's ``bind_tools`` /
        ``ToolNode`` accept it), while carrying the backing task so it resolves to
        itself on the worker.
        """

        flyte_task: typing.Any = None

        @property
        def task(self) -> typing.Any:
            return self.flyte_task

        @property
        def __wrapped_task__(self) -> typing.Any:
            return self.flyte_task
else:  # pragma: no cover - langchain-core missing
    FlyteStructuredTool = None  # type: ignore[assignment,misc]


def tool(
    func: AsyncFunctionTaskTemplate | typing.Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> typing.Any:
    """Convert a Flyte task (or plain callable) into a LangChain ``StructuredTool``.

    - For an ``@env.task``: returns a ``StructuredTool`` whose async body runs the
      task as a durable Flyte child action when the graph invokes it. The input
      schema is inferred from the task's typed signature. The backing task is
      wired to :class:`~flyteplugins.agents.core.ToolTaskResolver` and exposed via
      ``__wrapped_task__`` so it resolves to itself on the worker (no recursion).
    - For a plain (async) callable: returns a ``StructuredTool`` that runs it inline.

    The result is a first-class LangGraph tool — bind it to a model or hand it to
    :func:`~flyteplugins.agents.langgraph.tool_node` /
    :func:`~flyteplugins.agents.langgraph.ai_node`.

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
    tool_name = name or task.func.__name__
    desc = (description or task.func.__doc__ or f"Run {tool_name}").strip()

    # ``functools.wraps`` copies the task function's signature (via ``__wrapped__``)
    # so ``StructuredTool.from_function`` infers the correct args schema, while the
    # body dispatches to ``task.aio`` for durable execution. ``coerce_tool_args``
    # relaxes LLM int->float args so Flyte's type engine doesn't reject e.g.
    # ``amount_usd=42`` for a ``float`` param.
    @functools.wraps(task.func)
    async def _arun(**kwargs: typing.Any) -> str:
        result = await task.aio(**coerce_tool_args(task, kwargs or {}))
        return _as_content(result)

    _arun.__name__ = tool_name

    # Wire the shared resolver so the task resolves to itself on the worker.
    attach_tool_resolver(task)

    structured = FlyteStructuredTool.from_function(
        coroutine=_arun,
        name=tool_name,
        description=desc,
        flyte_task=task,
    )
    structured.__name__ = tool_name  # type: ignore[attr-defined]
    return structured


def _callable_to_tool(
    func: typing.Callable,
    *,
    name: str | None = None,
    description: str | None = None,
) -> typing.Any:
    """Build a LangChain ``StructuredTool`` from a plain callable."""
    from langchain_core.tools import StructuredTool as _StructuredTool

    tool_name = name or getattr(func, "__name__", "tool")
    desc = (description or func.__doc__ or f"Run {tool_name}").strip()

    @functools.wraps(func)
    async def _arun(**kwargs: typing.Any) -> str:
        out = func(**(kwargs or {}))
        if inspect.isawaitable(out):
            out = await out
        return _as_content(out)

    _arun.__name__ = tool_name

    structured = _StructuredTool.from_function(coroutine=_arun, name=tool_name, description=desc)
    structured.__name__ = tool_name  # type: ignore[attr-defined]
    return structured


def _as_content(result: typing.Any) -> str:
    """Convert a tool result to a string for LangChain's ``ToolMessage``."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)


def _coerce_tool(t: typing.Any) -> typing.Any:
    """Coerce a bare ``@env.task`` (or plain callable) into a LangChain tool.

    Already-wrapped tools (anything exposing ``ainvoke``) pass through unchanged.
    """
    if isinstance(t, AsyncFunctionTaskTemplate):
        return tool(t)
    if callable(t) and not hasattr(t, "ainvoke"):
        return tool(t)
    return t
