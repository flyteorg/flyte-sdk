"""Turn Flyte tasks into Pydantic AI tools that execute as durable actions.

Pydantic AI accepts plain (async) callables as tools in ``Agent(tools=[...])``
and derives each tool's JSON schema by inspecting the callable's signature and
docstring. :func:`tool` wraps a Flyte ``@env.task`` as one such callable: it
preserves the task's typed signature (via ``functools.wraps(task.func)``, which
also sets ``__wrapped__`` so Pydantic AI's schema inference sees the real
parameters) and dispatches to the task via ``task.aio()`` — so when the agent
calls the tool, it runs as a durable Flyte child action (its own
container/resources, with retries and caching) rather than inline in the agent's
process.
"""

from __future__ import annotations

import functools
import typing
from functools import partial

from flyte._task import AsyncFunctionTaskTemplate
from flyteplugins.agents.core import attach_tool_resolver, coerce_tool_args, task_json_schema


def tool(
    func: AsyncFunctionTaskTemplate | typing.Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> typing.Any:
    """Convert a Flyte task (or plain callable) into a Pydantic AI tool.

    - For an ``@env.task``: returns a plain async callable that Pydantic AI
      accepts natively in ``Agent(tools=[...])``. It carries the task's typed
      signature so Pydantic AI infers the correct input schema, and dispatches to
      ``task.aio()`` so each agent tool call runs as a durable Flyte child action.
      The backing task is wired to
      :class:`~flyteplugins.agents.core.ToolTaskResolver` and exposed via
      ``__wrapped_task__`` / ``.task`` so it resolves to itself on the worker
      (no recursion).
    - For a plain (async) callable: returned as-is (with ``name`` / ``description``
      overrides applied best-effort) — Pydantic AI inspects it directly.

    Usable bare, parametrized, or as a direct call::

        @tool
        @env.task
        async def get_weather(city: str) -> str: ...
    """
    if func is None:
        return partial(tool, name=name, description=description)
    if isinstance(func, AsyncFunctionTaskTemplate):
        return _task_to_tool(func, name=name, description=description)
    if not callable(func):
        raise TypeError(f"tool() expects a Flyte @env.task or a callable, got {type(func).__name__!r}.")
    return _relabel_callable(func, name=name, description=description)


def _task_to_tool(
    task: AsyncFunctionTaskTemplate,
    *,
    name: str | None = None,
    description: str | None = None,
) -> typing.Any:
    """Build a Pydantic AI tool callable from a Flyte task.

    Returns an async function that:
    - passes ``inspect.isfunction()`` and carries the task's typed signature
      (via ``functools.wraps``) so Pydantic AI infers the right input schema;
    - dispatches to ``task.aio()`` (a durable Flyte child action), relaxing
      LLM ``int``->``float`` args via ``coerce_tool_args``;
    - exposes ``__wrapped_task__`` and ``.task`` for the resolver/conformance.
    """
    inner = task.func
    task_json_schema(task)  # validate the task's input schema at construction time

    @functools.wraps(inner)
    async def _wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # In a Flyte task context this submits a durable child action; locally it runs
        # inline. ``functools.wraps`` keeps ``inner``'s signature (and ``__wrapped__``)
        # so Pydantic AI builds the right tool schema; ``coerce_tool_args`` relaxes
        # LLM int->float args.
        return await task.aio(*args, **coerce_tool_args(task, kwargs))

    if name:
        _wrapper.__name__ = name
    if description:
        _wrapper.__doc__ = description

    # The wrapper shadows the task at module scope; expose the real task and wire the
    # shared resolver so it resolves to itself on the worker (no recursion).
    _wrapper.__wrapped_task__ = task  # type: ignore[attr-defined]
    _wrapper.task = task  # type: ignore[attr-defined]
    attach_tool_resolver(task)
    return typing.cast(typing.Any, _wrapper)


def _relabel_callable(
    func: typing.Callable,
    *,
    name: str | None = None,
    description: str | None = None,
) -> typing.Callable:
    """Apply ``name`` / ``description`` overrides to a plain callable, best-effort.

    Pydantic AI inspects the callable directly to build its tool schema, so the
    callable is returned usable as-is. A slotted/immutable callable that rejects
    attribute assignment simply keeps its original name/doc.
    """
    if name:
        try:
            func.__name__ = name  # type: ignore[attr-defined]
        except (AttributeError, TypeError):  # pragma: no cover - slotted/immutable callable
            pass
    if description:
        try:
            func.__doc__ = description
        except (AttributeError, TypeError):  # pragma: no cover - slotted/immutable callable
            pass
    return func


def _coerce_tool(t: typing.Any) -> typing.Any:
    """Coerce a tool to a Pydantic AI-compatible callable.

    A bare ``@env.task`` is wrapped via :func:`tool`; anything else (an already
    ``tool``-wrapped callable, or a native Pydantic AI ``Tool``) is returned as-is.
    """
    if isinstance(t, AsyncFunctionTaskTemplate):
        return tool(t)
    return t
