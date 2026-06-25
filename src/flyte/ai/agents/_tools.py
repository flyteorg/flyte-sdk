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

from flyte._internal.resolvers.default import DefaultTaskResolver
from flyte.types._json_coercion import coerce_json_args

from ._llm import LLMCallable

if TYPE_CHECKING:
    from flyte._task import TaskTemplate
    from flyte.remote._task import LazyEntity


_ToolExecutor = Callable[[dict[str, Any]], Awaitable[Any]]

# A ``call_handler`` wraps how a tool is invoked. It is called as::
#
#     async def handler(call_llm: LLMCallable, tool_fn: ToolFn, **kwargs) -> Any:
#         ...
#
# where ``call_llm`` is the owning agent's LLM callback, ``tool_fn`` is the
# :class:`ToolFn` for the tool being invoked (call it to run the default
# behavior, or reach into ``tool_fn.target`` to customize), and ``**kwargs`` are
# the arguments the model produced for the call. Whatever the handler returns is
# used as the tool result.
ToolCallHandler = Callable[..., Awaitable[Any]]


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
    - inject a fully custom JSON schema,
    - intercept invocation with a ``call_handler``.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    execute: _ToolExecutor
    requires_approval: bool = False
    source: Literal["function", "task", "trace", "remote_task", "mcp", "custom"] = "function"
    # The underlying object this tool wraps (``@env.task`` template, plain
    # callable, ``LazyEntity``, …) when one exists. Custom-built / MCP tools
    # leave this ``None``. Exposed to ``call_handler`` via ``ToolFn.target``.
    target: Any = None
    # Optional interceptor that customizes how the tool is invoked. See
    # :data:`ToolCallHandler`.
    call_handler: ToolCallHandler | None = None

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

    @property
    def __wrapped_task__(self) -> Any:
        """The underlying ``TaskTemplate`` when this tool wraps one, else ``None``.

        When ``@tool`` is stacked on ``@env.task`` the module attribute becomes
        this :class:`AgentTool`, shadowing the task. Flyte's task resolver looks
        for this attribute to recover the real task for remote execution.
        """
        from flyte._task import TaskTemplate

        return self.target if isinstance(self.target, TaskTemplate) else None


@dataclass
class ToolFn:
    """The tool under invocation, handed to a :data:`ToolCallHandler`.

    Awaiting the instance runs the tool's *default* behavior::

        result = await tool_fn(**kwargs)

    The attributes give a custom handler everything it needs to change that
    behavior without re-deriving it. The most useful are:

    - :attr:`target` — the underlying ``@env.task`` template, plain callable, or
      ``LazyEntity`` (``None`` for custom / MCP tools). Reach into it to, e.g.,
      ``tool_fn.target.override(resources=...).aio(**kwargs)``.
    - :attr:`model` — the owning agent's model id, to pass to ``call_llm`` when
      the handler wants to consult the LLM.
    - :attr:`name` / :attr:`description` / :attr:`parameters` — the tool's
      LLM-facing metadata.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    model: str
    target: Any
    source: str
    _execute: _ToolExecutor

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the tool's default behavior with keyword arguments."""
        if args:
            raise TypeError(f"Tool '{self.name}' takes keyword arguments only; got positional arguments {args!r}.")
        return await self._execute(kwargs)


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


def _native_interface_for_target(target: Any) -> Any | None:
    """Return a :class:`~flyte.models.NativeInterface` for *target*, if derivable."""
    if target is None:
        return None
    from flyte._task import TaskTemplate
    from flyte.models import NativeInterface

    if isinstance(target, TaskTemplate):
        return target.native_interface
    if callable(target):
        fn = getattr(target, "__wrapped__", target)
        try:
            return NativeInterface.from_callable(fn)
        except Exception:
            return None
    return None


async def _coerce_tool_args(target: Any, args: dict[str, Any]) -> dict[str, Any]:
    """Coerce LLM JSON tool arguments using the wrapped callable's type hints."""
    iface = _native_interface_for_target(target)
    if iface is None:
        return args
    return await coerce_json_args(args, iface.inputs)


def _kwargs_from_call(target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Merge positional and keyword sandbox arguments into a kwargs dict."""
    fn: Any = target
    from flyte._task import TaskTemplate

    if isinstance(target, TaskTemplate):
        fn = getattr(target, "func", None)
    elif callable(target):
        fn = getattr(target, "__wrapped__", target)
    if fn is None:
        if args:
            raise TypeError(f"Tool target does not accept positional arguments; got {args!r}")
        return dict(kwargs)
    try:
        bound = inspect.signature(fn).bind_partial(*args, **kwargs)
    except TypeError as exc:
        raise TypeError(f"Invalid arguments for tool call: {exc}") from exc
    bound.apply_defaults()
    return dict(bound.arguments)


async def invoke_agent_tool(
    tool: AgentTool,
    args: dict[str, Any],
    *,
    call_llm: LLMCallable,
    model: str,
) -> Any:
    """Run *tool*, routing through ``call_handler`` when one is registered."""
    if tool.call_handler is not None:
        coerced_args = await _coerce_tool_args(tool.target, args)
        bound = ToolFn(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            model=model,
            target=tool.target,
            source=tool.source,
            _execute=tool.execute,
        )
        return await tool.call_handler(call_llm, bound, **coerced_args)
    return await tool.execute(args)


async def invoke_agent_tool_from_call(
    tool: AgentTool,
    *args: Any,
    call_llm: LLMCallable,
    model: str,
    **kwargs: Any,
) -> Any:
    """Like :func:`invoke_agent_tool` but accepts a Monty-style ``*args, **kwargs`` call."""
    call_args = _kwargs_from_call(tool.target, args, kwargs)
    return await invoke_agent_tool(tool, call_args, call_llm=call_llm, model=model)


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
        coerced = await _coerce_tool_args(fn, args)
        if is_async:
            return await fn(**coerced)
        return await asyncio.to_thread(fn, **coerced)

    return AgentTool(
        name=actual_name,
        description=_callable_short_doc(getattr(fn, "__wrapped__", fn)) or f"Execute {actual_name}",
        parameters=_json_schema_for_callable(getattr(fn, "__wrapped__", fn)),
        execute=execute,
        source="trace" if is_trace else "function",
        target=fn,
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
        coerced = await _coerce_tool_args(task, args)
        return await task.aio(**coerced)

    return AgentTool(
        name=actual_name,
        description=description,
        parameters=parameters,
        execute=execute,
        source="task",
        target=task,
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
        target=lazy,
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
    call_handler: ToolCallHandler | None = ...,
) -> Callable[[Any], AgentTool]: ...


def tool(
    obj: Any = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    requires_approval: bool = False,
    call_handler: ToolCallHandler | None = None,
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

    Pass ``call_handler`` to intercept *how* the tool is invoked. The handler is
    an async callback ``(call_llm, tool_fn, **kwargs) -> result`` that runs in
    place of the default execution. ``tool_fn`` is a :class:`ToolFn`: await it to
    run the default behavior, or use ``tool_fn.target`` (the underlying task /
    callable) and ``call_llm`` to do something custom — e.g. ask the LLM how to
    size compute, then run the task with overridden resources and retry on OOM::

        async def right_size(call_llm, tool_fn, **kwargs):
            resources = await _ask_llm_for_resources(call_llm, tool_fn, kwargs)
            return await tool_fn.target.override(resources=resources).aio(**kwargs)

        @tool(call_handler=right_size)
        @env.task
        async def train(...): ...

    Args:
        obj: The object to wrap. Omit when using ``tool`` as a parametrized
            decorator (``@tool(...)``).
        name: Override the tool name shown to the model. Defaults to the
            function / task name.
        description: Override the description shown to the model. Defaults to
            the first paragraph of the object's docstring.
        requires_approval: Gate execution behind the agent's HITL approval
            callback.
        call_handler: Optional async interceptor ``(call_llm, tool_fn, **kwargs)``
            that customizes how the tool is invoked. See :data:`ToolCallHandler`
            and :class:`ToolFn`.

    Returns:
        An :class:`AgentTool` (direct call) or a decorator returning one.
    """

    def _wrap(target: Any) -> AgentTool:
        base = _to_agent_tool(target, name=name)
        result = replace(
            base,
            description=description if description is not None else base.description,
            requires_approval=requires_approval,
            call_handler=call_handler if call_handler is not None else base.call_handler,
        )
        # Stacking ``@tool`` on ``@env.task`` rebinds the module attribute to this
        # tool, shadowing the task. Attach a resolver that recovers the task on
        # the worker so the *default* resolver stays untouched.
        _attach_tool_task_resolver(result.target)
        return result

    # Bare ``@tool`` usage passes the decorated object positionally; the
    # parametrized ``@tool(...)`` / keyword usage defers until the target is
    # supplied.
    if obj is None:
        return _wrap
    return _wrap(obj)


# ----------------------------------------------------------------------------
# Resolving ``@tool``-wrapped tasks for remote execution
# ----------------------------------------------------------------------------


class ToolTaskResolver(DefaultTaskResolver):
    """Resolver for a task shadowed at module scope by an ``@tool`` wrapper.

    Stacking ``@tool`` on ``@env.task`` rebinds the module attribute to the
    resulting :class:`AgentTool`, so the default resolver's ``getattr`` returns
    the tool rather than the :class:`~flyte._task.TaskTemplate`. This resolver
    recovers the underlying task via the wrapper's ``__wrapped_task__`` hook.

    ``@tool`` attaches an instance of this to the wrapped task's
    ``task_resolver`` (see :func:`_attach_tool_task_resolver`), so the default
    resolver is left completely untouched. Loader-arg generation is inherited
    unchanged from :class:`DefaultTaskResolver`.
    """

    @property
    def import_path(self) -> str:
        return "flyte.ai.agents._tools.ToolTaskResolver"

    def load_task(self, loader_args):  # type: ignore[override]
        from flyte._task import TaskTemplate

        task_def = super().load_task(loader_args)
        if isinstance(task_def, TaskTemplate):
            return task_def
        unwrapped = getattr(task_def, "__wrapped_task__", None)
        if isinstance(unwrapped, TaskTemplate):
            return unwrapped
        return task_def


def _attach_tool_task_resolver(target: Any) -> None:
    """Point a wrapped task at :class:`ToolTaskResolver` so it resolves remotely.

    Only applies to ``@env.task`` async-function templates that don't already
    declare a custom resolver; everything else (plain callables, ``LazyEntity``,
    user-supplied resolvers) is left untouched.
    """
    from flyte._task import AsyncFunctionTaskTemplate

    if isinstance(target, AsyncFunctionTaskTemplate) and target.task_resolver is None:
        target.task_resolver = ToolTaskResolver()


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


async def _stringify_tool_result(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    from flyte.types._json_coercion import serialize_json_value

    serialized = await serialize_json_value(result)
    try:
        return json.dumps(serialized, default=str)
    except (TypeError, ValueError):
        return str(result)


def _registry_uses_flyte_io(registry: Mapping[str, AgentTool]) -> bool:
    """Return True when any registered tool accepts or exposes flyte.io blob types."""

    def _schema_has_io(schema: Any) -> bool:
        if not isinstance(schema, dict):
            return False
        if schema.get("format") in ("blob", "structured-dataset"):
            return True
        for prop in schema.get("properties", {}).values():
            if _schema_has_io(prop):
                return True
        if _schema_has_io(schema.get("items")):
            return True
        for variant in schema.get("oneOf", []):
            if _schema_has_io(variant):
                return True
        return False

    return any(_schema_has_io(tool.parameters) for tool in registry.values())


def _abbreviate(value: Any, *, max_chars: int = 500) -> str:
    from flyte._utils.asyn import run_sync

    text = run_sync(_stringify_tool_result, value) if not isinstance(value, str) else value
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [+{len(text) - max_chars} chars]"
