"""Turn Flyte tasks into CrewAI tools that execute as durable actions.

CrewAI requires tools attached to an ``Agent(tools=[...])`` to be
``crewai.tools.BaseTool`` instances — plain callables are rejected by pydantic
validation. :func:`tool` therefore wraps a Flyte ``@env.task`` as a ``BaseTool``
subclass whose execution dispatches to the task via ``task.aio()`` — so when the
agent calls the tool, it runs as a durable Flyte child action (its own
container/resources, with retries and caching) rather than inline in the agent's
process.

Sync/async bridge: CrewAI invokes tools synchronously (``BaseTool.run`` ->
``_run``; the agent loop routes through ``CrewStructuredTool.invoke``, which calls
``self.func`` — our ``_run`` — and, if it returns a coroutine, ``asyncio.run``s it).
``asyncio.run`` explodes inside the already-running loop of a Flyte task, so we make
``_run`` a *synchronous* method that bridges to ``task.aio()`` via
:func:`flyte._utils.asyn.run_sync`, which drives the coroutine on a dedicated
background-thread loop and works from within a running loop. ``_arun`` awaits the
task directly for CrewAI's native async path.
"""

from __future__ import annotations

import functools
import inspect
import json
import typing
from functools import partial

from flyte._task import AsyncFunctionTaskTemplate
from flyte._utils.asyn import run_sync
from flyte.models import NativeInterface
from flyteplugins.agents.core import attach_tool_resolver, coerce_tool_args, task_json_schema


def tool(
    func: AsyncFunctionTaskTemplate | typing.Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> typing.Any:
    """Convert a Flyte task (or plain callable) into a CrewAI ``BaseTool``.

    - For an ``@env.task``: returns a ``BaseTool`` whose execution runs the task as
      a durable Flyte child action when the agent invokes it. The input schema is
      derived from the task via the Flyte type engine. The backing task is wired
      to :class:`~flyteplugins.agents.core.ToolTaskResolver` and exposed via
      ``__wrapped_task__`` so it resolves to itself on the worker (no recursion).
    - For a plain (async) callable: returns a ``BaseTool`` that runs it inline.

    The returned object is a native ``crewai.tools.BaseTool`` instance, so it can be
    attached directly to ``Agent(tools=[...])``.

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


def _args_model_from_signature(fn: typing.Callable, model_name: str) -> type:
    """Build a pydantic model describing ``fn``'s parameters for CrewAI's args schema.

    CrewAI derives a tool's args schema from ``BaseTool.args_schema`` (or, failing
    that, from the ``_run`` signature). Our ``_run`` takes ``**kwargs``, so we hand
    CrewAI an explicit model built from the wrapped callable's annotations.
    """
    from pydantic import create_model

    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # pragma: no cover - unresolved annotations
        hints = {}
    fields: dict[str, typing.Any] = {}
    for pname, param in inspect.signature(fn).parameters.items():
        if pname in ("self", "return"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        fallback = param.annotation if param.annotation is not inspect.Parameter.empty else typing.Any
        annotation = hints.get(pname, fallback)
        if param.default is inspect.Parameter.empty:
            fields[pname] = (annotation, ...)
        else:
            fields[pname] = (annotation, param.default)
    return create_model(model_name, **fields)


def _make_base_tool_class() -> type:
    """Define the ``FlyteCrewAITool`` ``BaseTool`` subclass (imported lazily).

    CrewAI is an optional heavy import, so the class is built on first use rather
    than at module import time.
    """
    from crewai.tools import BaseTool

    class FlyteCrewAITool(BaseTool):
        """A CrewAI ``BaseTool`` backed by a Flyte task (or plain callable).

        ``_run`` is synchronous by design: CrewAI's structured-tool path calls it
        and ``asyncio.run``s any returned coroutine, which would fail inside a
        Flyte task's running loop. We instead bridge to the async dispatcher via
        :func:`run_sync` (a background-thread loop) and return a plain string.
        """

        # Pydantic model config: allow the non-field private attributes below.
        _dispatch: typing.Callable[..., typing.Awaitable[typing.Any]]
        _wrapped_task: typing.Any

        def _run(self, **kwargs: typing.Any) -> str:
            return run_sync(self._dispatch, **kwargs)

        async def _arun(self, **kwargs: typing.Any) -> str:
            return await self._dispatch(**kwargs)

        @property
        def __wrapped_task__(self) -> typing.Any:
            return self._wrapped_task

        @property
        def task(self) -> typing.Any:
            return self._wrapped_task

    return FlyteCrewAITool


@functools.lru_cache(maxsize=1)
def _base_tool_class() -> type:
    return _make_base_tool_class()


def _build_tool(
    *,
    tool_name: str,
    desc: str,
    args_model: type,
    dispatch: typing.Callable[..., typing.Awaitable[typing.Any]],
    wrapped_task: typing.Any,
) -> typing.Any:
    cls = _base_tool_class()
    instance = cls(name=tool_name, description=desc, args_schema=args_model)
    # Private attrs live outside pydantic validation; set them directly.
    object.__setattr__(instance, "_dispatch", dispatch)
    object.__setattr__(instance, "_wrapped_task", wrapped_task)
    return instance


def _task_to_tool(
    task: AsyncFunctionTaskTemplate,
    *,
    name: str | None = None,
    description: str | None = None,
) -> typing.Any:
    """Build a CrewAI ``BaseTool`` from a Flyte task."""
    tool_name = name or task.func.__name__
    desc = (description or task.func.__doc__ or f"Run {tool_name}").strip()
    task_json_schema(task)  # validate schema at construction time
    args_model = _args_model_from_signature(task.func, f"{tool_name}_args")

    async def _dispatch(**kwargs: typing.Any) -> str:
        # In a Flyte task context this submits a durable child action; locally it
        # runs inline. ``coerce_tool_args`` relaxes LLM int->float args so Flyte's
        # type engine doesn't reject e.g. ``amount_usd=42`` for a ``float`` param.
        result = await task.aio(**coerce_tool_args(task, kwargs or {}))
        return _as_content(result)

    # Wire the shared resolver so the task resolves to itself on the worker.
    attach_tool_resolver(task)

    return _build_tool(
        tool_name=tool_name,
        desc=desc,
        args_model=args_model,
        dispatch=_dispatch,
        wrapped_task=task,
    )


def _callable_to_tool(
    func: typing.Callable,
    *,
    name: str | None = None,
    description: str | None = None,
) -> typing.Any:
    """Build a CrewAI ``BaseTool`` from a plain callable."""
    tool_name = name or getattr(func, "__name__", "tool")
    desc = (description or func.__doc__ or f"Run {tool_name}").strip()
    NativeInterface.from_callable(func).json_schema  # validate schema at construction time
    args_model = _args_model_from_signature(func, f"{tool_name}_args")

    async def _dispatch(**kwargs: typing.Any) -> str:
        out = func(**(kwargs or {}))
        if inspect.isawaitable(out):
            out = await out
        return _as_content(out)

    return _build_tool(
        tool_name=tool_name,
        desc=desc,
        args_model=args_model,
        dispatch=_dispatch,
        wrapped_task=None,
    )


def _as_content(result: typing.Any) -> str:
    """Convert a tool result to a string for CrewAI."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)


def _coerce_tool(t: typing.Any) -> typing.Any:
    """Coerce a tool to a CrewAI-compatible ``BaseTool``.

    Bare ``@env.task`` templates are wrapped on the fly; already-wrapped tools
    (``BaseTool`` instances) and other objects pass through unchanged.
    """
    if isinstance(t, AsyncFunctionTaskTemplate):
        return tool(t)
    return t
