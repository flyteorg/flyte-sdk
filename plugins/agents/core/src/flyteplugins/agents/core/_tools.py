"""Make a tool-backing Flyte task resolve to itself on the worker.

Every adapter stacks its ``function_tool`` on top of ``@env.task``, which rebinds
the module attribute to the tool and shadows the task. Without a guard, the
worker's default resolver loads the tool, the task runner calls the tool's
``execute``, and the task re-dispatches itself — recursing without end.

:class:`ToolTaskResolver` recovers the real task via the tool's
``__wrapped_task__`` hook; :func:`attach_tool_resolver` wires it onto the task.
Each adapter's tool object only has to expose ``__wrapped_task__`` returning its
backing :class:`~flyte._task.TaskTemplate`.
"""

from __future__ import annotations

import functools
import typing
from functools import partial

from flyte._internal.resolvers.default import DefaultTaskResolver
from flyte._task import AsyncFunctionTaskTemplate, TaskTemplate


class ToolTaskResolver(DefaultTaskResolver):
    """Resolver for a task shadowed at module scope by a tool wrapper.

    Recovers the underlying task via the wrapper's ``__wrapped_task__`` hook so
    the worker runs the task's own body instead of re-dispatching the tool.
    """

    @property
    def import_path(self) -> str:
        return "flyteplugins.agents.core._tools.ToolTaskResolver"

    def load_task(self, loader_args):  # type: ignore[override]
        task_def = super().load_task(loader_args)
        if isinstance(task_def, TaskTemplate):
            return task_def
        unwrapped = getattr(task_def, "__wrapped_task__", None)
        if isinstance(unwrapped, TaskTemplate):
            return unwrapped
        return task_def


def attach_tool_resolver(task: typing.Any) -> None:
    """Point a tool-backing ``@env.task`` at :class:`ToolTaskResolver`.

    No-op for anything that isn't an async-function task or that already declares
    a custom resolver, so the default resolver is left untouched elsewhere.
    """
    if isinstance(task, AsyncFunctionTaskTemplate) and task.task_resolver is None:
        task.task_resolver = ToolTaskResolver()


def coerce_tool_args(task: AsyncFunctionTaskTemplate, kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Coerce LLM-supplied tool args to the task's declared parameter types.

    LLMs emit JSON numbers without a fractional part as ``int`` (e.g. ``42``), but
    Flyte's type engine rejects an ``int`` where a ``float`` is declared — so the tool
    call fails during input conversion, before the child action is even submitted (the
    action never appears in the UI, and the task body never runs). This converts ``int``
    -> ``float`` for float-annotated params (leaving ``bool`` alone). Best-effort: returns
    the kwargs unchanged if the task's annotations can't be resolved.
    """
    try:
        hints = typing.get_type_hints(task.func)
    except Exception:  # pragma: no cover - unresolved annotations; pass through
        return kwargs
    coerced = dict(kwargs)
    for name, value in kwargs.items():
        if hints.get(name) is float and isinstance(value, int) and not isinstance(value, bool):
            coerced[name] = float(value)
    return coerced


def function_tool(
    func: AsyncFunctionTaskTemplate | typing.Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> typing.Callable:
    """Wrap a Flyte ``@env.task`` as a plain async tool function — the generic default.

    For SDKs that accept plain Python callables as tools (deriving the schema from the
    signature + docstring), this is the whole adapter ``function_tool``: the returned
    function carries the task's signature (``functools.wraps``), dispatches to
    ``task.aio()`` (so each call is a durable Flyte child action), exposes
    ``__wrapped_task__``, and wires the backing task to :class:`ToolTaskResolver`.
    Adapters whose SDK needs a native tool type (e.g. OpenAI's
    ``FunctionTool``, Claude's MCP ``SdkMcpTool``) provide their own instead.

    Usable bare, parametrized or as a direct call::

        @function_tool
        @env.task
        async def get_weather(city: str) -> str: ...
    """
    if func is None:
        return partial(function_tool, name=name, description=description)
    if isinstance(func, AsyncFunctionTaskTemplate):
        return _task_to_tool(func, name=name, description=description)
    # A plain function is already usable as a tool as-is.
    return func


def _task_to_tool(
    task: AsyncFunctionTaskTemplate,
    *,
    name: str | None = None,
    description: str | None = None,
) -> typing.Callable:
    inner = task.func

    @functools.wraps(inner)
    async def tool(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # In a Flyte task context this submits a durable child action; locally it runs
        # inline. ``functools.wraps`` keeps ``inner``'s signature so the SDK builds the
        # right tool declaration; ``coerce_tool_args`` relaxes LLM int->float args.
        return await task.aio(*args, **coerce_tool_args(task, kwargs))

    if name:
        tool.__name__ = name
    if description:
        tool.__doc__ = description

    # The wrapper shadows the task at module scope; expose the real task and wire the
    # shared resolver so it resolves to itself on the worker (no recursion).
    tool.__wrapped_task__ = task  # type: ignore[attr-defined]
    tool.task = task  # type: ignore[attr-defined]
    attach_tool_resolver(task)
    return tool


def task_json_schema(task: AsyncFunctionTaskTemplate) -> dict[str, typing.Any]:
    """The JSON schema of a Flyte task's inputs, via the Flyte type engine.

    Useful for adapters whose SDK wants a JSON-schema tool definition and that
    prefer Flyte's type-engine schema (correct ``Literal`` enums, ``File``/``Dir``
    /``DataFrame``, dataclasses) over the SDK's own signature inspection.
    """
    return task.json_schema
