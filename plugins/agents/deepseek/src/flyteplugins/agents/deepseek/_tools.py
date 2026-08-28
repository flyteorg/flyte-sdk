"""Turn Flyte tasks into DeepSeek Harness tools that execute as durable actions.

Unlike the client-side SDKs, DeepSeek Harness does not accept Python callables as
tools: the model loop and its whole tool surface live inside the harness runtime
subprocess, composed from Cordis plugins (bash, string editor, ...). The Python
SDK's wire protocol (`initialize` / `session/prompt`) has no tool-registration
message, so there is nothing to hand a Python function to.

What the harness *does* give every composition is local bash in a workspace
directory we choose. So a Flyte-task tool is published into that workspace as a
small executable shim, and the shim calls back into this process (see
`._bridge`) to run the real task. From the model's point of view it is an
ordinary command it can run; from Flyte's point of view the call is a durable
child action with its own container/resources, retries and caching — the same
guarantee every other adapter's `tool` provides.

`HarnessTool` is that published tool: a name, a description, a JSON schema
derived from the task via the Flyte type engine, and the coroutine the bridge
dispatches to.
"""

from __future__ import annotations

import inspect
import json
import typing
from functools import partial

from flyte._task import AsyncFunctionTaskTemplate
from flyte.models import NativeInterface
from flyteplugins.agents.core import attach_tool_resolver, coerce_tool_args, task_json_schema


class HarnessTool:
    """A tool the DeepSeek Harness runtime can call, backed by a Flyte task.

    The adapter publishes one bash shim per tool into the harness workspace and
    dispatches the shim's callback to `invoke`.
    """

    def __init__(
        self,
        name: str,
        description: str,
        schema: dict[str, typing.Any],
        handler: typing.Callable[[dict[str, typing.Any]], typing.Awaitable[typing.Any]],
    ) -> None:
        self.name = name
        self.description = description
        self.schema = schema
        self._handler = handler

    async def invoke(self, args: dict[str, typing.Any]) -> str:
        """Run the tool and render its result as text for the model."""
        return _as_text(await self._handler(args or {}))

    def usage(self) -> str:
        """The line the model is shown for this tool, in its instructions."""
        params = self.schema.get("properties") or {}
        required = set(self.schema.get("required") or ())
        rendered = ", ".join(
            f"{key}: {_type_name(spec)}" + ("" if key in required else " (optional)") for key, spec in params.items()
        )
        return f"{self.name}({rendered}) — {self.description}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"HarnessTool(name={self.name!r})"


def tool(
    func: AsyncFunctionTaskTemplate | typing.Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> HarnessTool | typing.Callable:
    """Convert a Flyte task (or plain callable) into a DeepSeek Harness tool.

    - For an `@env.task`: returns a `HarnessTool` that runs the task as a durable
      Flyte child action when the harness calls it. The input schema is derived
      from the task via the Flyte type engine. The backing task is wired to
      `flyteplugins.agents.core.ToolTaskResolver` and exposed via
      `__wrapped_task__` so it resolves to itself on the worker (no recursion).
    - For a plain (async) callable: returns a `HarnessTool` that runs it inline.

    Usable bare, parametrized, or as a direct call:

    ```python
    @tool
    @env.task
    async def get_weather(city: str) -> str: ...
    ```
    """
    if func is None:
        return partial(tool, name=name, description=description)
    if isinstance(func, AsyncFunctionTaskTemplate):
        return _task_to_tool(func, name=name, description=description)
    if not callable(func):
        raise TypeError(f"tool() expects a Flyte @env.task or a callable, got {type(func).__name__!r}.")
    return _callable_to_tool(func, name=name, description=description)


def _task_to_tool(
    task: AsyncFunctionTaskTemplate,
    *,
    name: str | None = None,
    description: str | None = None,
) -> HarnessTool:
    tool_name = name or task.func.__name__
    desc = (description or task.func.__doc__ or f"Run {tool_name}").strip()

    async def handler(args: dict[str, typing.Any]) -> typing.Any:
        # In a Flyte task context this submits a durable child action; locally it
        # runs inline. ``coerce_tool_args`` relaxes LLM int->float args so Flyte's
        # type engine doesn't reject e.g. ``amount_usd=42`` for a ``float`` param.
        return await task.aio(**coerce_tool_args(task, args or {}))

    harness_tool = HarnessTool(tool_name, desc, task_json_schema(task), handler)

    # The tool shadows the task at module scope, so wire the shared resolver and
    # expose the real task for it to recover on the worker.
    attach_tool_resolver(task)
    harness_tool.__wrapped_task__ = task  # type: ignore[attr-defined]
    harness_tool.task = task  # type: ignore[attr-defined]
    return harness_tool


def _callable_to_tool(
    func: typing.Callable,
    *,
    name: str | None = None,
    description: str | None = None,
) -> HarnessTool:
    tool_name = name or getattr(func, "__name__", "tool")
    desc = (description or func.__doc__ or f"Run {tool_name}").strip()
    schema = NativeInterface.from_callable(func).json_schema

    async def handler(args: dict[str, typing.Any]) -> typing.Any:
        out = func(**(args or {}))
        if inspect.isawaitable(out):
            out = await out
        return out

    return HarnessTool(tool_name, desc, schema, handler)


def _type_name(spec: typing.Any) -> str:
    """A short type label for a property in the tool's JSON schema."""
    if not isinstance(spec, dict):
        return "any"
    if "enum" in spec:
        return " | ".join(json.dumps(value) for value in spec["enum"])
    declared = spec.get("type")
    if isinstance(declared, list):
        return " | ".join(str(entry) for entry in declared)
    return str(declared or "any")


def _as_text(result: typing.Any) -> str:
    """Render a tool result as the text the harness reads back from the shim."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):  # pragma: no cover - exotic result type
        return str(result)
