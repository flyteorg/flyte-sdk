"""Turn Flyte tasks into OpenAI Agents tools that execute as durable actions."""

from __future__ import annotations

import inspect
import json
import typing
from dataclasses import dataclass
from functools import partial

from agents import FunctionTool as OpenAIFunctionTool
from agents import function_tool as openai_function_tool
from agents.function_schema import function_schema
from agents.tool_context import ToolContext
from flyte._task import AsyncFunctionTaskTemplate, TaskTemplate
from flyte.models import NativeInterface
from flyteplugins.agents.core import attach_tool_resolver


@dataclass
class FunctionTool(OpenAIFunctionTool):
    """An OpenAI Agents ``FunctionTool`` backed by a Flyte task.

    Behaves exactly like ``agents.FunctionTool`` from the SDK's perspective, but
    when the agent invokes it the call is dispatched to the underlying Flyte task
    — so it runs as a durable child action (its own container/resources, with
    retries and caching) rather than inline in the agent's process.
    """

    task: TaskTemplate | None = None
    native_interface: NativeInterface | None = None
    report: bool = False

    async def execute(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        """Run the wrapped task directly (a durable child action in a task context)."""
        if self.task is None:  # pragma: no cover - defensive
            raise RuntimeError("FunctionTool has no backing task to execute.")
        return await self.task.aio(*args, **kwargs)

    @property
    def __wrapped_task__(self) -> typing.Any:
        """The backing :class:`TaskTemplate` so :class:`ToolTaskResolver` can recover it.

        Stacking ``@tool`` on ``@env.task`` rebinds the module attribute
        to this tool, hiding the task from the default resolver;
        ``flyteplugins.agents.core.ToolTaskResolver`` uses this hook to recover the
        real task on the worker (so the task runs its own body instead of
        re-dispatching itself, which would otherwise recurse indefinitely).
        """
        return self.task if isinstance(self.task, TaskTemplate) else None


def tool(
    func: AsyncFunctionTaskTemplate | typing.Callable | None = None,
    **kwargs: typing.Any,
) -> FunctionTool | OpenAIFunctionTool:
    """Flyte-aware replacement for ``agents.function_tool`` — named ``tool`` for consistency.

    - For an ``@env.task`` (an ``AsyncFunctionTaskTemplate``): returns a
      :class:`FunctionTool` whose invocation runs the task as a durable Flyte
      action. The tool's JSON schema, name and description are derived by the
      OpenAI Agents SDK from the task's function signature, so strict-mode tool
      calling works unchanged.
    - For a plain callable or a ``@flyte.trace`` helper: forwards to the native
      ``agents.function_tool`` (runs inline; ``@flyte.trace`` helpers are still
      recorded for observability when inside a task).

    ``**kwargs`` (e.g. ``name_override``, ``description_override``) are forwarded
    to ``agents.function_tool`` in both cases.

    Usable as a bare decorator, a parametrized decorator, or a direct call::

        @tool
        @env.task
        async def get_weather(city: str) -> str: ...

        weather = tool(get_weather, name_override="weather")
    """
    if func is None:
        return partial(tool, **kwargs)

    if isinstance(func, AsyncFunctionTaskTemplate):
        return _task_to_tool(func, **kwargs)

    # Plain callables and @flyte.trace-decorated functions use the native path.
    return openai_function_tool(func, **kwargs)


def _task_to_tool(task: AsyncFunctionTaskTemplate, **kwargs: typing.Any) -> FunctionTool:
    """Build a :class:`FunctionTool` from a Flyte task.

    Only the stable, public fields of ``agents.FunctionTool`` are copied from a
    base tool built by ``agents.function_tool`` — we deliberately do not reflect
    over private fields, so this stays robust across SDK versions.
    """
    base = openai_function_tool(task.func, **kwargs)

    async def _on_invoke_tool(ctx: ToolContext[typing.Any], input: str) -> typing.Any:
        data: dict[str, typing.Any] = json.loads(input) if input else {}
        schema = function_schema(task.func)
        parsed = schema.params_pydantic_model(**data) if data else schema.params_pydantic_model()
        args, kwargs_ = schema.to_call_args(parsed)
        # In a Flyte task context, calling the task submits a durable child
        # action through the controller; locally it runs inline. Sync tasks
        # return a value directly, async tasks return an awaitable.
        out = task(*args, **kwargs_)
        if inspect.isawaitable(out):
            out = await out
        return out

    # The returned tool shadows the task at module scope, so point the task at the
    # shared resolver that recovers it on the worker via ``__wrapped_task__``.
    attach_tool_resolver(task)

    return FunctionTool(
        name=base.name,
        description=base.description,
        params_json_schema=base.params_json_schema,
        on_invoke_tool=_on_invoke_tool,
        strict_json_schema=base.strict_json_schema,
        task=task,
        native_interface=task.native_interface,
        report=task.report,
    )
