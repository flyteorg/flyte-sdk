"""Agent — a flyte-native tool-use agent harness.

The harness operates a simple, robust LLM tool-call loop:

1. Send the conversation + tool catalog to the LLM.
2. If the assistant returns tool calls, execute each one (sequentially or
   concurrently), append the results back into the message history, and loop.
3. Stop when the assistant returns a plain text reply (no tool calls) or when
   ``max_turns`` is reached.

Design goals
------------

- **Minimal core**: a single agent loop with clear extension points.
- **Tools = anything callable**: plain functions, ``@flyte.trace`` helpers,
  ``@env.task`` :class:`~flyte.TaskTemplate` instances (durable, on-cluster
  execution), ``LazyEntity`` references to remote tasks, or pre-built
  :class:`AgentTool` instances.
- **MCP integration**: connect to external MCP servers (Slack, GitHub, Linear,
  filesystem, etc.) and surface their tools to the agent transparently.
- **Memory**: optional :class:`MemoryStore` that serializes conversation
  history plus path-addressed artifacts to / from :class:`flyte.io.Dir`,
  letting the agent persist state across runs (e.g. across scheduled wake-ups
  or webhook invocations). Includes opt-in audit log, read-only prefixes, and
  optimistic concurrency for multi-agent / sleep-wake patterns.
- **HITL**: opt-in per-tool human approval that pauses the loop and waits for a
  human via the ``flyteplugins-hitl`` plugin before executing the tool.
- **Flyte-native**: implements the :class:`AgentProtocol` so it works
  seamlessly with :class:`~flyte.ai.chat.AgentChatAppEnvironment` and is happy
  to be wrapped in ``@env.task(triggers=...)`` for scheduled or webhook-driven
  wake-ups.

Heavy inspiration is taken from the `pi <https://github.com/earendil-works/pi>`_
agent harness — in particular its event model and the separation of the loop
from message conversion / tool invocation.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import json
import logging
import pathlib
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Literal,
    Mapping,
    Sequence,
    cast,
)

import flyte

from .memory import (
    AccessDenied,
    ConcurrencyError,
    MemoryMeta,
    MemoryStore,
    MemoryStoreError,
)
from .protocol import AgentResult

# Re-export memory types so existing imports of
# ``from flyte.ai.agents.agent import MemoryStore`` (and friends) keep working.
__all__ = [
    "AccessDenied",
    "ConcurrencyError",
    "MemoryMeta",
    "MemoryStore",
    "MemoryStoreError",
]

if TYPE_CHECKING:
    from flyte._task import TaskTemplate
    from flyte.remote._task import LazyEntity

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Progress hook (parallel to the codemode one)
# ----------------------------------------------------------------------------

AgentProgressCallback = Callable[["AgentEvent"], Awaitable[None]]

agent_progress_cb: contextvars.ContextVar[AgentProgressCallback | None] = contextvars.ContextVar(
    "agent_progress_cb",
    default=None,
)


async def _emit(event: "AgentEvent") -> None:
    cb = agent_progress_cb.get()
    if cb is None:
        return
    try:
        await cb(event)
    except Exception:  # pragma: no cover - progress hooks must never break the loop
        logger.exception("Agent progress callback raised; suppressing")


# ----------------------------------------------------------------------------
# Event + tool + memory dataclasses
# ----------------------------------------------------------------------------


EventType = Literal[
    "agent_start",
    "agent_end",
    "turn_start",
    "turn_end",
    "message",
    "tool_start",
    "tool_end",
    "tool_error",
    "approval_request",
    "approval_decision",
]


@dataclass
class AgentEvent:
    """Lightweight event emitted by the agent loop.

    The agent stays decoupled from any specific UI: subscribe via
    :data:`agent_progress_cb` to forward these to logs, NDJSON streams, websockets,
    Flyte reports, etc.
    """

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)


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
# MCP server spec
# ----------------------------------------------------------------------------


@dataclass(kw_only=True)
class MCPServerSpec:
    """Declarative spec for a remote MCP server that exposes tools.

    The agent connects on startup, lists available tools, and registers each as
    a callable tool whose ``execute`` proxies the MCP ``tools/call`` request.

    Either ``url`` (for HTTP/SSE/streamable-http transports) or ``command``
    (for stdio transports) must be set.

    Parameters
    ----------
    name:
        Stable display name for logs and event payloads.
    url:
        HTTP(S) URL of the MCP endpoint (e.g. ``https://host/mcp/mcp``).
    command:
        Command to launch a stdio MCP server (e.g.
        ``["uvx", "mcp-server-github"]``).
    headers:
        Optional HTTP headers (for ``Authorization`` etc.).
    env:
        Optional environment variables for stdio launches.
    transport:
        Transport hint. ``"auto"`` (default) infers from ``url`` / ``command``.
    tool_prefix:
        Optional prefix prepended to each tool name to avoid collisions.
    tool_filter:
        Optional allowlist of tool names to expose. ``None`` means all.
    """

    name: str
    url: str | None = None
    command: list[str] | None = None
    headers: dict[str, str] | None = None
    env: dict[str, str] | None = None
    transport: Literal["auto", "http", "streamable-http", "sse", "stdio"] = "auto"
    tool_prefix: str = ""
    tool_filter: list[str] | None = None

    def __post_init__(self) -> None:
        if not self.url and not self.command:
            raise ValueError("MCPServerSpec requires either `url` or `command`.")


# ----------------------------------------------------------------------------
# Tool resolution
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
        if isinstance(obj, AgentTool):
            tool = obj
            if override_name:
                tool = AgentTool(
                    name=override_name,
                    description=tool.description,
                    parameters=tool.parameters,
                    execute=tool.execute,
                    requires_approval=tool.requires_approval,
                    source=tool.source,
                )
        elif _is_task_template(obj):
            tool = _make_task_tool(obj, name=override_name)
        elif _is_lazy_entity(obj):
            tool = _make_lazy_entity_tool(obj, name=override_name)
        elif callable(obj):
            tool = _make_callable_tool(obj, name=override_name)
        else:
            raise TypeError(
                f"Cannot turn {type(obj).__name__!r} into an AgentTool. "
                "Pass an AgentTool, a callable, a @flyte.trace helper, an "
                "@env.task template, a LazyEntity, or a {name: object} mapping."
            )

        if tool.name in resolved:
            raise ValueError(f"Duplicate tool name '{tool.name}'")
        resolved[tool.name] = tool
    return resolved


# ----------------------------------------------------------------------------
# Skills
# ----------------------------------------------------------------------------


def _load_skills(skills: Sequence[str | pathlib.Path]) -> str:
    parts: list[str] = []
    for skill in skills:
        if isinstance(skill, pathlib.Path):
            parts.append(skill.read_text())
        else:
            parts.append(skill)
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------
# LLM callback (litellm based by default)
# ----------------------------------------------------------------------------


@dataclass
class LLMMessage:
    """Provider-agnostic shape returned by :data:`LLMCallable`.

    ``tool_calls`` follows the OpenAI tool-calling convention; provider-specific
    callers should normalize to this shape.
    """

    content: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: Any = None


LLMCallable = Callable[
    [str, str, list[dict[str, Any]], list[dict[str, Any]] | None],
    Awaitable[LLMMessage],
]


async def _default_call_llm(
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> LLMMessage:
    """Default LLM callback that uses ``litellm.acompletion`` with tool calling.

    Compatible with any provider that litellm supports (OpenAI, Anthropic,
    Gemini, Bedrock, local OpenAI-compatible servers, …).
    """
    try:
        from litellm import acompletion
    except ImportError as exc:  # pragma: no cover - exercised by integration tests only
        raise ImportError(
            "litellm is not installed. Install with `pip install litellm` "
            "or pass `call_llm=...` with a custom callback."
        ) from exc

    full_messages: list[dict[str, Any]] = [{"role": "system", "content": system}, *messages]
    kwargs: dict[str, Any] = {"model": model, "messages": full_messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = await acompletion(**kwargs)
    choice = response.choices[0]  # type: ignore[index]
    msg = choice.message
    tool_calls: list[dict[str, Any]] = []
    for call in getattr(msg, "tool_calls", None) or []:
        try:
            args_str = call.function.arguments
            args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
        except json.JSONDecodeError:
            args = {"_raw": call.function.arguments}
        tool_calls.append(
            {
                "id": getattr(call, "id", None) or f"call_{uuid.uuid4().hex[:12]}",
                "name": call.function.name,
                "arguments": args,
            }
        )
    return LLMMessage(content=getattr(msg, "content", None) or "", tool_calls=tool_calls, raw=response)


# ----------------------------------------------------------------------------
# HITL hook
# ----------------------------------------------------------------------------


ApprovalCallback = Callable[[AgentTool, dict[str, Any]], Awaitable[bool]]


async def _hitl_approval(tool: AgentTool, args: dict[str, Any]) -> bool:
    """Default HITL approval: ask the user via ``flyteplugins-hitl``.

    Returns ``True`` if the user approves the tool call. When the plugin is not
    installed (or the agent is running outside a Flyte task context), this
    falls back to denying the call so that the agent can recover by trying a
    different approach.
    """
    try:
        import flyteplugins.hitl as hitl
    except ImportError:
        logger.warning(
            "Tool %s requires approval but `flyteplugins-hitl` is not installed; denying by default.",
            tool.name,
        )
        return False

    pretty_args = json.dumps(args, indent=2, default=str)
    prompt = f"Approve tool call `{tool.name}`?\n\nArguments:\n{pretty_args}"
    event = await hitl.new_event.aio(
        f"approve_{tool.name}_{uuid.uuid4().hex[:6]}",
        data_type=bool,
        scope="run",
        prompt=prompt,
    )
    decision = await event.wait.aio()
    return bool(decision)


# ----------------------------------------------------------------------------
# MCP tool loader (lazy / optional)
# ----------------------------------------------------------------------------


class _MCPToolLoader:
    """Discovers tools from an MCP server and surfaces them as :class:`AgentTool`.

    Stays inactive until :meth:`load` is called. We delay all MCP imports here
    so that ``Agent`` itself has no required dependency on the ``mcp``
    package.
    """

    def __init__(self, specs: Sequence[MCPServerSpec]):
        self.specs = list(specs)
        self._sessions: list[Any] = []

    async def load(self) -> list[AgentTool]:
        if not self.specs:
            return []
        try:
            from mcp import ClientSession  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "MCP servers configured but the `mcp` package is not installed. "
                "Install with `pip install mcp` or `pip install 'flyte[mcp]'`."
            ) from exc

        tools: list[AgentTool] = []
        for spec in self.specs:
            tools.extend(await self._load_one(spec))
        return tools

    async def _load_one(self, spec: MCPServerSpec) -> list[AgentTool]:
        if spec.command:
            return await self._load_stdio(spec)
        return await self._load_http(spec)

    async def _load_stdio(self, spec: MCPServerSpec) -> list[AgentTool]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        assert spec.command is not None
        params = StdioServerParameters(command=spec.command[0], args=spec.command[1:], env=spec.env)

        @contextlib.asynccontextmanager
        async def _session() -> AsyncIterator[Any]:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session

        return await self._materialize(spec, _session)

    async def _load_http(self, spec: MCPServerSpec) -> list[AgentTool]:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        url = spec.url
        assert url is not None
        headers = spec.headers

        @contextlib.asynccontextmanager
        async def _session() -> AsyncIterator[Any]:
            async with streamablehttp_client(url, headers=headers) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session

        return await self._materialize(spec, _session)

    async def _materialize(
        self,
        spec: MCPServerSpec,
        session_cm: Callable[[], "contextlib.AbstractAsyncContextManager[Any]"],
    ) -> list[AgentTool]:
        async with session_cm() as session:
            listing = await session.list_tools()

        tools: list[AgentTool] = []
        for raw_tool in listing.tools:  # type: ignore[attr-defined]
            short_name = getattr(raw_tool, "name", None)
            if not short_name:
                continue
            if spec.tool_filter is not None and short_name not in spec.tool_filter:
                continue
            tool_name = f"{spec.tool_prefix}{short_name}" if spec.tool_prefix else short_name

            description = getattr(raw_tool, "description", "") or f"MCP tool from {spec.name}"
            input_schema = getattr(raw_tool, "inputSchema", None) or {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            }

            async def _execute(args: dict[str, Any], *, _short_name: str = short_name) -> Any:
                async with session_cm() as inner_session:
                    result = await inner_session.call_tool(_short_name, arguments=args)
                content = getattr(result, "content", None)
                if not content:
                    return None
                if isinstance(content, list):
                    parts: list[Any] = []
                    for chunk in content:
                        text = getattr(chunk, "text", None)
                        parts.append(text if text is not None else chunk)
                    return "\n".join(str(p) for p in parts) if all(isinstance(p, str) for p in parts) else parts
                return content

            tools.append(
                AgentTool(
                    name=tool_name,
                    description=description,
                    parameters=input_schema,
                    execute=_execute,
                    source="mcp",
                )
            )
        return tools


# ----------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------


@dataclass(kw_only=True)
class Agent:
    """A flyte-native tool-use agent harness.

    Parameters
    ----------
    name:
        Stable agent identifier (used for logs and event payloads).
    instructions:
        Base system prompt. Skills and a tool catalog summary are appended
        automatically.
    model:
        Model identifier passed to ``call_llm``. Defaults to
        ``"claude-haiku-4-5"`` to match :class:`CodeModeAgent`.
    tools:
        Sequence (or ``{name: tool}`` mapping) of tools the agent may call.
        Each entry can be a plain callable, a ``@flyte.trace`` helper, an
        ``@env.task`` :class:`~flyte.TaskTemplate`, a
        :class:`~flyte.remote._task.LazyEntity`, or a pre-built
        :class:`AgentTool`.
    mcp_servers:
        Optional remote / stdio MCP servers whose tools should be loaded into
        the agent's tool registry on first use. See :class:`MCPServerSpec`.
    skills:
        Extra context appended to the system prompt. Each entry is either a
        string or a :class:`pathlib.Path` pointing to a local text file.
    max_turns:
        Maximum number of LLM ↔ tool turns before the loop gives up.
    call_llm:
        Optional async callback ``(model, system, messages, tools) -> LLMMessage``.
        Defaults to :func:`_default_call_llm` (uses litellm).
    memory:
        Optional :class:`MemoryStore` initialized from a previous session.
        When provided, the existing transcript is prepended to every
        conversation and the in-flight transcript is appended to it on each
        call. Tools may also use the same instance for path-addressed
        artifact reads / writes (with audit + optional concurrency).
    approval_callback:
        Optional async callback ``(tool, args) -> bool`` invoked when a tool
        with ``requires_approval=True`` is about to run. Defaults to a HITL
        prompt via ``flyteplugins-hitl``.
    parallel_tool_calls:
        When ``True`` (default) tool calls returned in a single assistant
        message are executed concurrently. Set to ``False`` to force strict
        sequential execution (useful when tool side-effects must be ordered).
    """

    name: str = "flyte-agent"
    instructions: str = "You are a helpful assistant."
    model: str = "claude-haiku-4-5"
    tools: Sequence[Any] | Mapping[str, Any] = field(default_factory=tuple)
    mcp_servers: Sequence[MCPServerSpec] = field(default_factory=tuple)
    skills: Sequence[str | pathlib.Path] = field(default_factory=tuple)
    max_turns: int = 25
    call_llm: LLMCallable = field(default=_default_call_llm)
    memory: MemoryStore | None = None
    approval_callback: ApprovalCallback = field(default=_hitl_approval)
    parallel_tool_calls: bool = True

    _registry: dict[str, AgentTool] = field(init=False, repr=False, default_factory=dict)
    _mcp_loaded: bool = field(init=False, repr=False, default=False)
    _mcp_loader: _MCPToolLoader = field(init=False, repr=False, default=None)  # type: ignore[assignment]
    _system_prompt: str = field(init=False, repr=False, default="")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self._registry = _resolve_tools(self.tools)
        self._mcp_loader = _MCPToolLoader(self.mcp_servers)
        self._system_prompt = self._build_system_prompt()

    # ------------------------------------------------------------------
    # Public introspection helpers
    # ------------------------------------------------------------------

    @property
    def system_prompt(self) -> str:
        """The fully-rendered system prompt, including skills + tool catalog."""
        return self._system_prompt

    def tool_descriptions(self) -> list[dict[str, str]]:
        """Conform to :class:`~flyte.ai.agents.protocol.Agent`.

        MCP tools loaded lazily are only listed after the first
        :meth:`run` call.
        """
        return [
            {
                "name": tool.name,
                "signature": _summarize_signature(tool),
                "description": tool.description,
            }
            for tool in self._registry.values()
        ]

    def add_tool(self, obj: Any, *, name: str | None = None) -> AgentTool:
        """Register an additional tool after construction.

        Useful when tools need access to runtime objects (e.g. an HTTP client
        created inside a task).
        """
        new = _resolve_tools({name: obj} if name else [obj])
        for tool in new.values():
            if tool.name in self._registry:
                raise ValueError(f"Duplicate tool name '{tool.name}'")
            self._registry[tool.name] = tool
        self._system_prompt = self._build_system_prompt()
        return next(iter(new.values()))

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        skills_block = ""
        if self.skills:
            skills_block = "\n\nAdditional context / skills:\n" + _load_skills(self.skills)

        tool_lines: list[str] = []
        for tool in self._registry.values():
            tool_lines.append(f"- {tool.name}: {tool.description}")
        tools_block = "\n".join(tool_lines) if tool_lines else "(no tools registered)"

        return (
            f"{self.instructions}\n\n"
            f"You have access to the following tools (full JSON schemas are provided "
            f"in the tool calling interface):\n"
            f"{tools_block}\n"
            f"Use tools deliberately. Reply with plain text when you have a final "
            f"answer or do not need a tool."
            f"{skills_block}"
        )

    # ------------------------------------------------------------------
    # Agent loop
    # ------------------------------------------------------------------

    async def _ensure_mcp_loaded(self) -> None:
        if self._mcp_loaded or not self.mcp_servers:
            self._mcp_loaded = True
            return
        loaded = await self._mcp_loader.load()
        for tool in loaded:
            if tool.name in self._registry:
                logger.warning("MCP tool name collision: %s. Skipping.", tool.name)
                continue
            self._registry[tool.name] = tool
        self._mcp_loaded = True
        self._system_prompt = self._build_system_prompt()

    def _llm_tools(self) -> list[dict[str, Any]] | None:
        if not self._registry:
            return None
        return [tool.to_openai_format() for tool in self._registry.values()]

    async def _execute_one(self, tool: AgentTool, args: dict[str, Any]) -> tuple[str, bool]:
        """Run a tool, applying optional HITL approval. Returns ``(stringified_result, is_error)``."""
        if tool.requires_approval:
            await _emit(AgentEvent("approval_request", {"tool": tool.name, "args": args}))
            approved = await self.approval_callback(tool, args)
            await _emit(AgentEvent("approval_decision", {"tool": tool.name, "approved": approved}))
            if not approved:
                return (f"Human reviewer declined tool `{tool.name}`. Try a different approach.", True)
        await _emit(AgentEvent("tool_start", {"tool": tool.name, "args": args}))
        try:
            result = await tool.execute(args)
        except Exception as exc:
            await _emit(AgentEvent("tool_error", {"tool": tool.name, "error": str(exc)}))
            return (f"Error executing tool `{tool.name}`: {exc}", True)
        await _emit(AgentEvent("tool_end", {"tool": tool.name, "result": _abbreviate(result)}))
        return (_stringify_tool_result(result), False)

    async def run(
        self,
        message: str,
        history: list[dict[str, Any]] | None = None,
    ) -> AgentResult:
        """Drive the LLM ↔ tool loop until the assistant returns a final reply.

        Implements the :class:`~flyte.ai.agents.protocol.Agent` protocol so
        instances can be plugged directly into
        :class:`~flyte.ai.chat.AgentChatAppEnvironment`.
        """
        await self._ensure_mcp_loaded()
        await _emit(AgentEvent("agent_start", {"name": self.name, "model": self.model}))
        t0 = time.monotonic()

        prior: list[dict[str, Any]] = []
        if self.memory is not None:
            prior.extend(self.memory.messages)
        if history:
            prior.extend(history)
        messages: list[dict[str, Any]] = [*prior, {"role": "user", "content": message}]

        tools_schema = self._llm_tools()
        attempts = 0
        last_text = ""
        error_msg = ""

        for turn in range(self.max_turns):
            attempts = turn + 1
            await _emit(AgentEvent("turn_start", {"turn": attempts, "max_turns": self.max_turns}))
            try:
                with flyte.group(f"{self.name}-turn-{attempts}"):
                    llm_msg = await self.call_llm(
                        self.model,
                        self._system_prompt,
                        list(messages),
                        tools_schema,
                    )
            except Exception as exc:
                error_msg = f"LLM call failed on turn {attempts}: {exc}"
                await _emit(AgentEvent("agent_end", {"error": error_msg, "turns": attempts}))
                if self.memory is not None:
                    self.memory.extend(messages[len(prior) :])
                return AgentResult(error=error_msg, attempts=attempts, summary=last_text)

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": llm_msg.content or ""}
            if llm_msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["arguments"]),
                        },
                    }
                    for call in llm_msg.tool_calls
                ]
            messages.append(assistant_msg)
            await _emit(AgentEvent("message", {"role": "assistant", "content": llm_msg.content or ""}))

            if not llm_msg.tool_calls:
                last_text = llm_msg.content or ""
                await _emit(
                    AgentEvent(
                        "turn_end",
                        {"turn": attempts, "had_tool_calls": False, "text_len": len(last_text)},
                    )
                )
                break

            tool_results = await self._execute_calls(llm_msg.tool_calls)
            for call, (result_text, _is_error) in zip(llm_msg.tool_calls, tool_results):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": call["name"],
                        "content": result_text,
                    }
                )

            await _emit(
                AgentEvent(
                    "turn_end",
                    {
                        "turn": attempts,
                        "had_tool_calls": True,
                        "tool_count": len(llm_msg.tool_calls),
                    },
                )
            )
        else:
            error_msg = f"Reached max_turns={self.max_turns} without producing a final answer."

        if self.memory is not None:
            self.memory.extend(messages[len(prior) :])

        elapsed = int((time.monotonic() - t0) * 1000)
        await _emit(
            AgentEvent(
                "agent_end",
                {"turns": attempts, "elapsed_ms": elapsed, "error": error_msg, "summary_len": len(last_text)},
            )
        )
        return AgentResult(summary=last_text, error=error_msg, attempts=attempts)

    async def _execute_calls(
        self,
        calls: list[dict[str, Any]],
    ) -> list[tuple[str, bool]]:
        """Run all tool calls in a batch, respecting ``parallel_tool_calls``."""
        if not self.parallel_tool_calls or len(calls) <= 1:
            results: list[tuple[str, bool]] = []
            for call in calls:
                results.append(await self._dispatch_call(call))
            return results

        return list(await asyncio.gather(*(self._dispatch_call(call) for call in calls)))

    async def _dispatch_call(self, call: dict[str, Any]) -> tuple[str, bool]:
        name = call["name"]
        args = call.get("arguments") or {}
        tool = self._registry.get(name)
        if tool is None:
            await _emit(AgentEvent("tool_error", {"tool": name, "error": "unknown tool"}))
            return (f"Unknown tool `{name}`.", True)
        if not isinstance(args, dict):
            args = {"_raw": args}
        return await self._execute_one(tool, args)


# ----------------------------------------------------------------------------
# Helpers used by the agent loop
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
