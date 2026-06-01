"""Tool resolution and serialization helpers for :class:`flyte.ai.agents.Agent`.

This module is internal: import the public symbols (``AgentTool``) from
:mod:`flyte.ai.agents` instead. The agent module re-exports the ``_``-prefixed
helpers below for callers that historically imported from
``flyte.ai.agents.agent``.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal, Mapping, Sequence, cast, overload

if TYPE_CHECKING:
    from flyte._task import TaskTemplate
    from flyte.remote._task import LazyEntity


_ToolExecutor = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass
class AgentTool:
    """A normalized tool descriptor used by :class:`Agent`.

    Most users do not construct :class:`AgentTool` directly — pass plain
    callables, ``@flyte.trace`` helpers, or ``@env.task`` templates to
    :class:`Agent` and they will be wrapped automatically. Build one
    explicitly when you need to:

    - rename a tool for the LLM,
    - override the description shown to the model,
    - require human approval before execution (HITL),
    - inject a fully custom JSON schema.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    execute: _ToolExecutor
    requires_approval: bool = False
    source: Literal["function", "task", "trace", "remote_task", "mcp", "custom"] = "function"

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to the OpenAI / litellm tools schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ----------------------------------------------------------------------------
# Type sniffers for the various objects we accept as tools
# ----------------------------------------------------------------------------


def _is_task_template(obj: Any) -> bool:
    from flyte._task import TaskTemplate

    return isinstance(obj, TaskTemplate)


def _is_lazy_entity(obj: Any) -> bool:
    try:
        from flyte.remote._task import LazyEntity
    except Exception:  # pragma: no cover
        return False
    return isinstance(obj, LazyEntity)


# ----------------------------------------------------------------------------
# Schema + doc extraction
# ----------------------------------------------------------------------------


def _callable_short_doc(fn: Callable[..., Any]) -> str:
    doc = inspect.getdoc(fn) or ""
    if not doc:
        return ""
    return doc.split("\n\n")[0].replace("\n", " ").strip()


def _json_schema_for_callable(fn: Callable[..., Any]) -> dict[str, Any]:
    """Best-effort JSON schema for a plain Python callable.

    Falls back to the Flyte type engine via ``NativeInterface`` for type-rich
    schemas (Literals, dataclasses, etc.), and degrades to a permissive
    ``object`` schema when extraction fails.
    """
    from flyte.models import NativeInterface

    try:
        return NativeInterface.from_callable(fn).json_schema
    except Exception:
        return {"type": "object", "properties": {}, "additionalProperties": True}


# ----------------------------------------------------------------------------
# Constructors for each kind of input
# ----------------------------------------------------------------------------


def _make_callable_tool(fn: Callable[..., Any], *, name: str | None = None) -> AgentTool:
    actual_name: str = name or getattr(fn, "__name__", None) or "tool"
    is_trace = bool(getattr(fn, "__wrapped__", None))
    is_async = inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(getattr(fn, "__wrapped__", fn))

    async def execute(args: dict[str, Any]) -> Any:
        if is_async:
            return await fn(**args)
        return await asyncio.to_thread(fn, **args)

    return AgentTool(
        name=actual_name,
        description=_callable_short_doc(getattr(fn, "__wrapped__", fn)) or f"Execute {actual_name}",
        parameters=_json_schema_for_callable(getattr(fn, "__wrapped__", fn)),
        execute=execute,
        source="trace" if is_trace else "function",
    )


def _make_task_tool(task: "TaskTemplate", *, name: str | None = None) -> AgentTool:
    underlying = cast(Any, task).func
    actual_name = name or underlying.__name__
    description = _callable_short_doc(underlying) or f"Execute Flyte task `{actual_name}`"

    parameters: dict[str, Any]
    try:
        parameters = task.json_schema  # type: ignore[attr-defined]
    except Exception:
        parameters = _json_schema_for_callable(underlying)

    async def execute(args: dict[str, Any]) -> Any:
        return await task.aio(**args)

    return AgentTool(
        name=actual_name,
        description=description,
        parameters=parameters,
        execute=execute,
        source="task",
    )


def _make_lazy_entity_tool(lazy: "LazyEntity", *, name: str | None = None) -> AgentTool:
    actual_name = name or lazy.name.rsplit("/", maxsplit=1)[-1]

    async def execute(args: dict[str, Any]) -> Any:
        return await lazy.aio(**args)  # type: ignore[attr-defined]

    return AgentTool(
        name=actual_name,
        description=f"Remote Flyte task `{lazy.name}`",
        parameters={"type": "object", "properties": {}, "additionalProperties": True},
        execute=execute,
        source="remote_task",
    )


def _to_agent_tool(obj: Any, *, name: str | None = None) -> AgentTool:
    """Normalize a single tool-like object into an :class:`AgentTool`.

    Accepts already-constructed :class:`AgentTool` instances, plain Python
    callables (sync or async), ``@flyte.trace`` helpers, ``@env.task``
    :class:`~flyte.TaskTemplate` instances, and
    :class:`~flyte.remote._task.LazyEntity` remote-task references.
    """
    if isinstance(obj, AgentTool):
        if name and name != obj.name:
            return replace(obj, name=name)
        return obj
    if _is_task_template(obj):
        return _make_task_tool(obj, name=name)
    if _is_lazy_entity(obj):
        return _make_lazy_entity_tool(obj, name=name)
    if callable(obj):
        return _make_callable_tool(obj, name=name)
    raise TypeError(
        f"Cannot turn {type(obj).__name__!r} into an AgentTool. "
        "Pass an AgentTool, a callable, a @flyte.trace helper, an "
        "@env.task template, a LazyEntity, or a {name: object} mapping."
    )


def _resolve_tools(
    tools: Sequence[Any] | Mapping[str, Any],
) -> dict[str, AgentTool]:
    """Normalize the user-provided ``tools`` argument into ``{name: AgentTool}``.

    Accepts:
    - already-constructed :class:`AgentTool` instances
    - plain Python callables (sync or async)
    - ``@flyte.trace`` helpers
    - ``@env.task`` :class:`~flyte.TaskTemplate` instances
    - :class:`~flyte.remote._task.LazyEntity` remote-task references
    """
    items: list[tuple[str | None, Any]]
    if isinstance(tools, Mapping):
        items = [(k, v) for k, v in tools.items()]
    else:
        items = [(None, v) for v in tools]

    resolved: dict[str, AgentTool] = {}
    for override_name, obj in items:
        tool = _to_agent_tool(obj, name=override_name)
        if tool.name in resolved:
            raise ValueError(f"Duplicate tool name '{tool.name}'")
        resolved[tool.name] = tool
    return resolved


@overload
def tool(obj: Any, /) -> AgentTool: ...


@overload
def tool(
    *,
    name: str | None = ...,
    description: str | None = ...,
    requires_approval: bool = ...,
) -> Callable[[Any], AgentTool]: ...


def tool(
    obj: Any = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    requires_approval: bool = False,
) -> AgentTool | Callable[[Any], AgentTool]:
    """Wrap a task, ``@flyte.trace`` helper, plain callable, or ``LazyEntity`` as an :class:`AgentTool`.

    This removes the boilerplate of building an :class:`AgentTool` by hand
    (manually pulling the docstring, JSON schema, and writing a dict-args
    execution bridge) when you only need to tweak how a tool is presented to
    the model or gate it behind human approval.

    Use it as a direct call::

        refund_tool = tool(issue_refund, requires_approval=True)

    or as a (parametrized) decorator stacked on top of ``@env.task`` /
    ``@flyte.trace`` / a plain function::

        @tool(requires_approval=True)
        @env.task
        async def issue_refund(order_id: str, amount_usd: float) -> dict: ...

        @tool
        def search(query: str) -> str: ...

    The wrapped task is still registered with its :class:`~flyte.TaskEnvironment`
    and executes on-cluster via ``task.aio`` when the agent calls it.

    Args:
        obj: The object to wrap. Omit when using ``tool`` as a parametrized
            decorator (``@tool(...)``).
        name: Override the tool name shown to the model. Defaults to the
            function / task name.
        description: Override the description shown to the model. Defaults to
            the first paragraph of the object's docstring.
        requires_approval: Gate execution behind the agent's HITL approval
            callback.

    Returns:
        An :class:`AgentTool` (direct call) or a decorator returning one.
    """

    def _wrap(target: Any) -> AgentTool:
        base = _to_agent_tool(target, name=name)
        return replace(
            base,
            description=description if description is not None else base.description,
            requires_approval=requires_approval,
        )

    # Bare ``@tool`` usage passes the decorated object positionally; the
    # parametrized ``@tool(...)`` / keyword usage defers until the target is
    # supplied.
    if obj is None:
        return _wrap
    return _wrap(obj)


# ----------------------------------------------------------------------------
# Helpers used by the agent loop for display / logging
# ----------------------------------------------------------------------------


def _summarize_signature(tool: AgentTool) -> str:
    """Compact pseudo-signature derived from the tool's JSON schema."""
    props = tool.parameters.get("properties", {}) if isinstance(tool.parameters, dict) else {}
    required = set(tool.parameters.get("required", []) if isinstance(tool.parameters, dict) else [])
    parts: list[str] = []
    for pname, schema in props.items():
        type_hint = schema.get("type", "any") if isinstance(schema, dict) else "any"
        if pname in required:
            parts.append(f"{pname}: {type_hint}")
        else:
            parts.append(f"{pname}?: {type_hint}")
    return f"{tool.name}({', '.join(parts)})"


def _stringify_tool_result(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)


def _abbreviate(value: Any, *, max_chars: int = 500) -> str:
    text = _stringify_tool_result(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [+{len(text) - max_chars} chars]"
