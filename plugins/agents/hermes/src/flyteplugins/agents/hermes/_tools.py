"""Turn Flyte tasks into Hermes tools that execute as durable actions.

Hermes (the ``hermes-agent`` package) does not accept tool callables on the
agent object. Tools live in a process-global registry (``tools.registry``),
keyed by name and grouped into *toolsets*; an ``AIAgent`` exposes whatever its
``enabled_toolsets`` resolve to. :func:`tool` therefore does two things:

1. wraps the Flyte ``@env.task`` with the shared core wrapper
   (:func:`flyteplugins.agents.core.tool`) so a call dispatches to
   ``task.aio()`` — a durable Flyte child action (its own container/resources,
   with retries and caching) — and the backing task resolves to itself on the
   worker;
2. registers that wrapper in the Hermes tool registry under the
   :data:`FLYTE_TOOLSET` toolset, with an OpenAI-format schema derived from the
   task via the Flyte type engine.

``run_agent`` then scopes each built agent to exactly the requested tools via a
custom toolset (see ``_run``); a bring-your-own ``AIAgent`` opts in with
``enabled_toolsets=[FLYTE_TOOLSET]``.
"""

from __future__ import annotations

import inspect
import json
import typing
from functools import partial

from flyte._task import AsyncFunctionTaskTemplate
from flyte.models import NativeInterface
from flyteplugins.agents.core import task_json_schema
from flyteplugins.agents.core import tool as core_tool

FLYTE_TOOLSET = "flyte"
"""The Hermes toolset every Flyte ``tool`` registers under."""


def tool(
    func: AsyncFunctionTaskTemplate | typing.Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> typing.Any:
    """Convert a Flyte task (or plain callable) into a Hermes tool.

    - For an ``@env.task``: returns the shared core tool wrapper (a plain async
      function dispatching to the task as a durable Flyte child action, with
      ``__wrapped_task__`` and the resolver wired), *and* registers it in the
      Hermes tool registry under :data:`FLYTE_TOOLSET` so an ``AIAgent`` can
      call it by name. The input schema is derived from the task via the Flyte
      type engine.
    - For a plain (sync or async) callable: registers it as an inline Hermes
      tool, deriving the schema from its signature.

    Usable bare, parametrized, or as a direct call::

        @tool
        @env.task
        async def get_weather(city: str) -> str: ...
    """
    if func is None:
        return partial(tool, name=name, description=description)

    wrapper = core_tool(func, name=name, description=description)
    tool_name = getattr(wrapper, "__name__", None) or type(wrapper).__name__

    if isinstance(func, AsyncFunctionTaskTemplate):
        desc = (description or func.func.__doc__ or f"Run {tool_name}").strip()
        parameters = task_json_schema(func)
    else:
        desc = (description or getattr(func, "__doc__", None) or f"Run {tool_name}").strip()
        parameters = NativeInterface.from_callable(func).json_schema

    _register_hermes_tool(tool_name, desc, parameters, wrapper)
    try:
        wrapper.__hermes_registered__ = True
    except (AttributeError, TypeError):  # pragma: no cover - slotted/immutable callable
        pass
    return wrapper


def _register_hermes_tool(
    tool_name: str,
    description: str,
    parameters: dict[str, typing.Any],
    wrapper: typing.Callable,
) -> None:
    """Register ``wrapper`` in the Hermes tool registry under :data:`FLYTE_TOOLSET`.

    The handler receives the model's arguments as a dict (Hermes dispatches
    ``handler(args, **context)``) and returns a string for the model. Re-registering
    the same name replaces the previous entry, so module reloads are safe.
    """
    from tools.registry import registry  # hermes-agent's process-global tool registry

    async def _handler(args: dict[str, typing.Any] | None = None, **_: typing.Any) -> str:
        out = wrapper(**(args or {}))
        if inspect.isawaitable(out):
            out = await out
        return _as_content(out)

    registry.register(
        name=tool_name,
        toolset=FLYTE_TOOLSET,
        schema={
            "type": "function",
            "function": {"name": tool_name, "description": description, "parameters": parameters},
        },
        handler=_handler,
        is_async=True,
        description=description,
    )


def _as_content(result: typing.Any) -> str:
    """Convert a tool result to a string for the model."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)
