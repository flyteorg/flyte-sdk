"""Turn Flyte tasks into Claude Agent SDK tools that execute as durable actions.

The Claude Agent SDK exposes custom tools as in-process MCP tools (``@tool`` /
``SdkMcpTool``). :func:`function_tool` wraps a Flyte ``@env.task`` as one whose
handler dispatches to the task via ``task.aio()`` — so when Claude calls the
tool, it runs as a durable Flyte child action (its own container/resources, with
retries and caching) rather than inline in the agent process.
"""

from __future__ import annotations

import inspect
import json
import typing
from functools import partial

from claude_agent_sdk import SdkMcpTool
from claude_agent_sdk import tool as claude_tool
from flyte._task import AsyncFunctionTaskTemplate
from flyte.models import NativeInterface
from flyteplugins.agents.core import attach_tool_resolver, coerce_tool_args, task_json_schema


def function_tool(
    func: AsyncFunctionTaskTemplate | typing.Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> SdkMcpTool | typing.Callable:
    """Convert a Flyte task (or plain callable) into a Claude Agent SDK tool.

    - For an ``@env.task``: returns an ``SdkMcpTool`` whose handler runs the task
      as a durable Flyte child action when Claude calls it. The input schema is
      derived from the task via the Flyte type engine. The backing task is wired
      to :class:`~flyteplugins.agents.core.ToolTaskResolver` and exposed via
      ``__wrapped_task__`` so it resolves to itself on the worker (no recursion).
    - For a plain (async) callable: returns an ``SdkMcpTool`` that runs it inline.

    Usable bare, parametrized, or as a direct call::

        @function_tool
        @env.task
        async def get_weather(city: str) -> str: ...
    """
    if func is None:
        return partial(function_tool, name=name, description=description)
    if isinstance(func, AsyncFunctionTaskTemplate):
        return _task_to_tool(func, name=name, description=description)
    return _callable_to_tool(func, name=name, description=description)


def _task_to_tool(
    task: AsyncFunctionTaskTemplate,
    *,
    name: str | None = None,
    description: str | None = None,
) -> SdkMcpTool:
    tool_name = name or task.func.__name__
    desc = (description or task.func.__doc__ or f"Run {tool_name}").strip()

    async def handler(args: dict[str, typing.Any]) -> dict[str, typing.Any]:
        # In a Flyte task context this submits a durable child action; locally it
        # runs inline. ``coerce_tool_args`` relaxes LLM int->float args so Flyte's
        # type engine doesn't reject e.g. ``amount_usd=42`` for a ``float`` param.
        result = await task.aio(**coerce_tool_args(task, args or {}))
        return _as_content(result)

    sdk_tool = claude_tool(tool_name, desc, task_json_schema(task))(handler)

    # The tool shadows the task at module scope, so wire the shared resolver and
    # expose the real task for it to recover on the worker.
    attach_tool_resolver(task)
    sdk_tool.__wrapped_task__ = task
    sdk_tool.task = task
    return sdk_tool


def _callable_to_tool(
    func: typing.Callable,
    *,
    name: str | None = None,
    description: str | None = None,
) -> SdkMcpTool:
    tool_name = name or getattr(func, "__name__", "tool")
    desc = (description or func.__doc__ or f"Run {tool_name}").strip()
    schema = NativeInterface.from_callable(func).json_schema

    async def handler(args: dict[str, typing.Any]) -> dict[str, typing.Any]:
        out = func(**(args or {}))
        if inspect.isawaitable(out):
            out = await out
        return _as_content(out)

    return claude_tool(tool_name, desc, schema)(handler)


def _as_content(result: typing.Any) -> dict[str, typing.Any]:
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    return {"content": [{"type": "text", "text": text}]}
