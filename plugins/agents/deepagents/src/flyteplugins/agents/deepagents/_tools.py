"""Turn Flyte tasks into Deep Agents tools that execute as durable actions.

Deep Agents (LangChain's agent harness) accepts LangChain ``BaseTool`` instances
as tools — both on the main agent (``create_deep_agent(tools=[...])``) and on
subagents (``SubAgent(tools=[...])``). :func:`tool` wraps a Flyte ``@env.task``
as a LangChain ``StructuredTool`` whose async coroutine dispatches to the task
via ``task.aio()`` — so when the agent calls the tool, it runs as a durable
Flyte child action (its own container/resources, with retries and caching)
rather than inline in the agent's process.

The returned object is a real ``StructuredTool`` (a ``BaseTool``), so it drops
straight into ``create_deep_agent(tools=[...])`` or a subagent's tool list. It
additionally exposes ``__wrapped_task__`` and ``task`` (via direct attribute
assignment, which ``StructuredTool`` permits) and wires the backing task to
:class:`~flyteplugins.agents.core.ToolTaskResolver` so it resolves to itself on
the worker (no recursion).
"""

from __future__ import annotations

import inspect
import json
import typing
from functools import partial

from flyte._task import AsyncFunctionTaskTemplate
from flyteplugins.agents.core import attach_tool_resolver, coerce_tool_args


def tool(
    func: AsyncFunctionTaskTemplate | typing.Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> typing.Any:
    """Convert a Flyte task (or plain callable) into a LangChain ``StructuredTool``.

    - For an ``@env.task``: returns a ``StructuredTool`` whose async coroutine runs
      the task as a durable Flyte child action when the agent invokes it. The input
      schema is derived from the task's typed signature. The backing task is wired to
      :class:`~flyteplugins.agents.core.ToolTaskResolver` and exposed via
      ``__wrapped_task__`` so it resolves to itself on the worker (no recursion).
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

    async def _arun(**kwargs: typing.Any) -> str:
        # In a Flyte task context this submits a durable child action; locally it
        # runs inline. ``coerce_tool_args`` relaxes LLM int->float args so Flyte's
        # type engine doesn't reject e.g. ``amount_usd=42`` for a ``float`` param.
        result = await task.aio(**coerce_tool_args(task, kwargs or {}))
        return _as_content(result)

    # Wire the shared resolver so the task resolves to itself on the worker.
    attach_tool_resolver(task)

    # Derive an explicit args schema from the task's typed signature. The coroutine
    # above is ``**kwargs``-only, so LangChain's own inference would produce a single
    # ``kwargs`` object param — we build the real pydantic model instead.
    args_schema = _args_schema_from_callable(task.func, tool_name)

    structured = StructuredTool.from_function(
        func=None,
        coroutine=_arun,
        name=tool_name,
        description=desc,
        args_schema=args_schema,
    )

    # Expose the real task and a convenient ``task`` alias so callers/tests can reach
    # the backing task. ``StructuredTool`` is a pydantic model whose ``__setattr__``
    # rejects non-field names, so set them through ``object.__setattr__`` (dunders
    # like ``__wrapped_task__`` would bypass it, but keep both paths uniform).
    object.__setattr__(structured, "__wrapped_task__", task)
    object.__setattr__(structured, "task", task)
    return structured


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

    async def _arun(**kwargs: typing.Any) -> str:
        out = func(**(kwargs or {}))
        if inspect.isawaitable(out):
            out = await out
        return _as_content(out)

    args_schema = _args_schema_from_callable(func, tool_name)

    return StructuredTool.from_function(
        func=None,
        coroutine=_arun,
        name=tool_name,
        description=desc,
        args_schema=args_schema,
    )


def _args_schema_from_callable(func: typing.Callable, tool_name: str) -> typing.Any | None:
    """Build a pydantic ``args_schema`` from a callable's typed signature.

    Returns ``None`` (letting LangChain infer) if the signature can't be resolved.
    The model's fields mirror the callable's parameters, with annotations and
    defaults preserved so the LLM sees a correct tool schema.
    """
    from pydantic import create_model

    try:
        hints = typing.get_type_hints(func)
        sig = inspect.signature(func)
    except Exception:  # pragma: no cover - unresolved annotations; let LangChain infer
        return None

    fields: dict[str, typing.Any] = {}
    for pname, param in sig.parameters.items():
        if pname == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = hints.get(pname, typing.Any)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[pname] = (annotation, default)

    if not fields:
        return None
    return create_model(f"{tool_name}Args", **fields)


def _as_content(result: typing.Any) -> str:
    """Convert a tool result to a string for LangChain's ToolMessage."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)


def _coerce_tool(t: typing.Any) -> typing.Any:
    """Coerce a bare ``@env.task`` into a LangChain tool; pass everything else through."""
    if isinstance(t, AsyncFunctionTaskTemplate):
        return tool(t)
    return t
