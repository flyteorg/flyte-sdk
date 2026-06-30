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

Implementation note: the tool/MCP/LLM building blocks live in sibling modules
(:mod:`._tools`, :mod:`._mcp`, :mod:`._llm`); this module focuses on the
``Agent`` class and the loop.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import pathlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Mapping, Sequence

import flyte
from flyte.syncify import syncify

# Internal building blocks. All four ``_``-prefixed tool helpers are used
# inside ``Agent`` and remain addressable as ``flyte.ai.agents.agent.<name>``
# for callers (notably this repo's tests) that historically imported them
# from here.
from ._llm import LLMCallable, LLMMessage, _default_call_llm
from ._mcp import MCPServerSpec, _MCPToolLoader
from ._tools import (
    AgentTool,
    _abbreviate,
    _bind_tool_invocation_context,
    _registry_uses_flyte_io,
    _resolve_tools,
    _stringify_tool_result,
    _summarize_signature,
    invoke_agent_tool,
)
from .memory import (
    MemoryStore,
)
from .protocol import AgentResult

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
# Event dataclass
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


# ----------------------------------------------------------------------------
# Loop driver plumbing
#
# The tool loop and the code-mode loop share the same scaffolding (MCP load,
# start/end events, timing, message setup, the call-LLM-per-turn cadence, the
# max-turns guard, and memory finalization). They differ only in what happens
# *after* the LLM responds each turn. That per-turn behavior is supplied as a
# ``step`` callback returning a :class:`_TurnResult`; :meth:`Agent._run_loop`
# owns everything else and returns a :class:`_RunOutcome`.
# ----------------------------------------------------------------------------


@dataclass
class _TurnResult:
    """What a per-turn step decided: stop with a final answer, or keep looping."""

    done: bool
    final_text: str = ""


@dataclass
class _RunOutcome:
    """Terminal state of the agent loop, used to build an :class:`AgentResult`."""

    attempts: int
    last_text: str
    error_msg: str
    memory: MemoryStore | None


# One turn of work after the LLM responds: append messages, emit events, run
# tools / sandbox, and report whether the loop should stop.
_TurnStep = Callable[[LLMMessage, "list[dict[str, Any]]", int], Awaitable[_TurnResult]]


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
        ``"claude-haiku-4-5"``.
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
    approval_callback:
        Optional async callback ``(tool, args) -> bool`` invoked when a tool
        with ``requires_approval=True`` is about to run. Defaults to a HITL
        prompt via ``flyteplugins-hitl``.
    parallel_tool_calls:
        When ``True`` (default) tool calls returned in a single assistant
        message are executed concurrently. Set to ``False`` to force strict
        sequential execution (useful when tool side-effects must be ordered).
        Ignored in code mode.
    code_mode:
        When ``True`` the agent runs in *code mode*: instead of emitting JSON
        tool calls, the LLM writes a small Python program each turn that is
        executed in the Monty sandbox (``flyte.sandbox.orchestrate_local``) with
        the tools exposed as plain functions. The value of the program's last
        expression becomes the observation for the next turn; the loop ends when
        the LLM replies with plain text (no code block). This unlocks generated
        control flow (loops, ``flyte_map`` fan-out, intermediate aggregation)
        while still dispatching ``@env.task`` tools durably on-cluster. Tools
        with a ``call_handler`` run through that handler in code mode as well.
        Requires ``pydantic-monty`` in the runtime image. Note: per-tool HITL
        approval is not enforced in code mode, since tools are invoked from inside
        the sandbox rather than as discrete approved calls.
    """

    name: str = "flyte-agent"
    instructions: str = "You are a helpful assistant."
    model: str = "claude-haiku-4-5"
    tools: Sequence[Any] | Mapping[str, Any] = field(default_factory=tuple)
    mcp_servers: Sequence[MCPServerSpec] = field(default_factory=tuple)
    skills: Sequence[str | pathlib.Path] = field(default_factory=tuple)
    max_turns: int = 25
    call_llm: LLMCallable = field(default=_default_call_llm)
    approval_callback: ApprovalCallback = field(default=_hitl_approval)
    parallel_tool_calls: bool = True
    code_mode: bool = False

    _registry: dict[str, AgentTool] = field(init=False, repr=False, default_factory=dict)
    _mcp_loaded: bool = field(init=False, repr=False, default=False)
    _mcp_loader: _MCPToolLoader = field(init=False, repr=False, default=None)  # type: ignore[assignment]
    _system_prompt: str = field(init=False, repr=False, default="")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self._registry = {
            name: _bind_tool_invocation_context(tool, call_llm=self.call_llm, model=self.model)
            for name, tool in _resolve_tools(self.tools).items()
        }
        self._mcp_loader = _MCPToolLoader(self.mcp_servers)
        if self.code_mode and any(t.requires_approval for t in self._registry.values()):
            logger.warning(
                "Agent %s is in code_mode; per-tool HITL approval is not enforced because tools "
                "are invoked from inside the sandbox. requires_approval is ignored for: %s",
                self.name,
                ", ".join(t.name for t in self._registry.values() if t.requires_approval),
            )
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
        for _tool in new.values():
            if _tool.name in self._registry:
                raise ValueError(f"Duplicate tool name '{_tool.name}'")
            self._registry[_tool.name] = _bind_tool_invocation_context(_tool, call_llm=self.call_llm, model=self.model)
        self._system_prompt = self._build_system_prompt()
        return _bind_tool_invocation_context(next(iter(new.values())), call_llm=self.call_llm, model=self.model)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        skills_block = ""
        if self.skills:
            skills_block = "\n\nAdditional context / skills:\n" + _load_skills(self.skills)

        if self.code_mode:
            from ._code import build_code_system_prompt

            return build_code_system_prompt(self.instructions, self._registry, skills_block)

        tool_lines: list[str] = []
        for _tool in self._registry.values():
            tool_lines.append(f"- {_tool.name}: {_tool.description}")
        tools_block = "\n".join(tool_lines) if tool_lines else "(no tools registered)"

        io_hint = ""
        if _registry_uses_flyte_io(self._registry):
            io_hint = (
                "\n\nWhen a tool returns a flyte.io File, Dir, or DataFrame (a JSON object "
                "with a `uri` field), pass the **complete** tool-result JSON as the argument "
                "to downstream tools that accept that type. Continue calling tools until the "
                "user's request is fully satisfied before giving a final text answer."
            )

        return (
            f"{self.instructions}\n\n"
            f"You have access to the following tools (full JSON schemas are provided "
            f"in the tool calling interface):\n"
            f"{tools_block}\n"
            f"Use tools deliberately. Reply with plain text when you have a final "
            f"answer or do not need a tool."
            f"{io_hint}"
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
        for _tool in loaded:
            if _tool.name in self._registry:
                logger.warning("MCP tool name collision: %s. Skipping.", _tool.name)
                continue
            self._registry[_tool.name] = _tool
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
            result = await invoke_agent_tool(tool, args, call_llm=self.call_llm, model=self.model)
        except Exception as exc:
            await _emit(AgentEvent("tool_error", {"tool": tool.name, "error": str(exc)}))
            return (f"Error executing tool `{tool.name}`: {exc}", True)
        await _emit(AgentEvent("tool_end", {"tool": tool.name, "result": _abbreviate(result)}))
        result_text = await _stringify_tool_result(result)
        return (result_text, False)

    @syncify
    async def run(
        self,
        message: str,
        memory: list[dict[str, Any]] | MemoryStore | None = None,
    ) -> AgentResult:
        """Drive the LLM ↔ tool loop until the assistant returns a final reply.

        Implements the :class:`~flyte.ai.agents.protocol.AgentProtocol` so
        instances can be plugged directly into
        :class:`~flyte.ai.chat.AgentChatAppEnvironment`.

        The agent is decoupled from any persistent state: memory is passed in
        per call rather than attached to the agent. ``memory`` may be:

        - ``None``: a stateless, single-shot conversation.
        - a ``list[dict]``: prior messages to prepend (e.g. a chat ``history``).
          The returned :class:`AgentResult` carries no memory in this case.
        - a :class:`MemoryStore`: its transcript is prepended, the in-flight
          transcript is appended back to it, and it is returned on
          :attr:`AgentResult.memory`. Persistence is the caller's
          responsibility: call ``memory.save()`` (or ``.save.aio()``) after
          ``run`` to write the updated transcript back to its keyed remote path.

        Call synchronously via ``run(...)``; in async contexts use ``run.aio(...)``.
        """
        if self.code_mode:
            return await self._run_code_mode(message, memory)

        tools_schema = self._llm_tools()

        async def step(llm_msg: LLMMessage, messages: list[dict[str, Any]], attempts: int) -> _TurnResult:
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": llm_msg.content or ""}
            if llm_msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call.get("arguments") or {}),
                        },
                    }
                    for call in llm_msg.tool_calls
                ]
            messages.append(assistant_msg)
            await _emit(AgentEvent("message", {"role": "assistant", "content": llm_msg.content or ""}))

            if not llm_msg.tool_calls:
                text = llm_msg.content or ""
                await _emit(AgentEvent("turn_end", {"turn": attempts, "had_tool_calls": False, "text_len": len(text)}))
                return _TurnResult(done=True, final_text=text)

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
                    {"turn": attempts, "had_tool_calls": True, "tool_count": len(llm_msg.tool_calls)},
                )
            )
            return _TurnResult(done=False)

        outcome = await self._run_loop(message, memory, tools_schema=tools_schema, step=step)
        return AgentResult(
            summary=outcome.last_text,
            error=outcome.error_msg,
            attempts=outcome.attempts,
            memory=outcome.memory,
        )

    async def _run_code_mode(
        self,
        message: str,
        memory: list[dict[str, Any]] | MemoryStore | None = None,
    ) -> AgentResult:
        """Code-mode loop: each turn the LLM writes Python executed in the sandbox.

        The LLM's program runs via ``flyte.sandbox.orchestrate_local`` with the
        agent's tools exposed as functions; the value of its last expression is
        fed back as the next observation. The loop ends when the LLM responds
        with plain text (no code block), which becomes the final answer. Sandbox
        errors are surfaced back to the LLM so it can self-correct on the next
        turn (bounded by ``max_turns``).
        """
        import flyte.sandbox

        from ._code import build_sandbox_tools, extract_python_code

        sandbox_tools = build_sandbox_tools(self._registry, call_llm=self.call_llm, model=self.model)
        last_code = ""

        async def step(llm_msg: LLMMessage, messages: list[dict[str, Any]], attempts: int) -> _TurnResult:
            nonlocal last_code
            text = llm_msg.content or ""
            messages.append({"role": "assistant", "content": text})
            await _emit(AgentEvent("message", {"role": "assistant", "content": text}))

            code = extract_python_code(text)
            if not code:
                # No code block: the assistant has produced its final answer.
                await _emit(AgentEvent("turn_end", {"turn": attempts, "had_code": False, "text_len": len(text)}))
                return _TurnResult(done=True, final_text=text)

            last_code = code
            await _emit(AgentEvent("tool_start", {"tool": "<sandbox>", "code": code}))
            try:
                with flyte.group(f"{self.name}-sandbox-{attempts}"):
                    result = await flyte.sandbox.orchestrate_local(
                        code,
                        inputs={"_unused": 0},
                        tasks=sandbox_tools,
                    )
            except Exception as exc:
                await _emit(AgentEvent("tool_error", {"tool": "<sandbox>", "error": str(exc)}))
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your code raised an error:\n\n```\n{exc}\n```\n\n"
                            "Fix the code and try again, respecting the Monty sandbox restrictions."
                        ),
                    }
                )
                await _emit(AgentEvent("turn_end", {"turn": attempts, "had_code": True, "error": True}))
                return _TurnResult(done=False)

            observation = await _stringify_tool_result(result)
            await _emit(AgentEvent("tool_end", {"tool": "<sandbox>", "result": _abbreviate(result)}))
            messages.append(
                {
                    "role": "user",
                    "content": f"Execution result:\n{observation}" if observation else "Execution result: (no value)",
                }
            )
            await _emit(AgentEvent("turn_end", {"turn": attempts, "had_code": True}))
            return _TurnResult(done=False)

        outcome = await self._run_loop(message, memory, tools_schema=None, step=step, mode="code")
        return AgentResult(
            code=last_code,
            summary=outcome.last_text,
            error=outcome.error_msg,
            attempts=outcome.attempts,
            memory=outcome.memory,
        )

    def _init_messages(
        self,
        memory: list[dict[str, Any]] | MemoryStore | None,
        message: str,
    ) -> tuple[MemoryStore | None, int, list[dict[str, Any]]]:
        """Seed the message list from prior context and the new user message.

        Returns the backing :class:`MemoryStore` (if any), the number of prior
        messages (so the in-flight tail can be sliced off later), and the
        message list to drive the loop with.
        """
        store: MemoryStore | None = memory if isinstance(memory, MemoryStore) else None
        prior: list[dict[str, Any]] = []
        if store is not None:
            prior.extend(store.messages)
        elif isinstance(memory, list):
            prior.extend(memory)
        messages: list[dict[str, Any]] = [*prior, {"role": "user", "content": message}]
        return store, len(prior), messages

    async def _run_loop(
        self,
        message: str,
        memory: list[dict[str, Any]] | MemoryStore | None,
        *,
        tools_schema: list[dict[str, Any]] | None,
        step: _TurnStep,
        mode: str | None = None,
    ) -> _RunOutcome:
        """Shared driver for both the tool loop and the code-mode loop.

        Owns the scaffolding common to both: MCP loading, ``agent_start`` /
        ``agent_end`` events, timing, message setup, the call-LLM-per-turn
        cadence (with error handling), the ``max_turns`` guard, and memory
        finalization. The mode-specific per-turn behavior is delegated to
        *step*, which appends messages, emits its own events, runs tools or the
        sandbox, and signals via :class:`_TurnResult` whether to stop.

        ``memory`` is mutated in place (the in-flight transcript is appended back
        to a passed :class:`MemoryStore`) but never saved; persistence is the
        caller's responsibility.
        """
        await self._ensure_mcp_loaded()
        start_data: dict[str, Any] = {"name": self.name, "model": self.model}
        if mode is not None:
            start_data["mode"] = mode
        await _emit(AgentEvent("agent_start", start_data))
        t0 = time.monotonic()

        store, prior_len, messages = self._init_messages(memory, message)

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
                        list[dict[str, Any]](messages),
                        tools_schema,
                    )
            except Exception as exc:
                error_msg = f"LLM call failed on turn {attempts}: {exc}"
                break

            outcome = await step(llm_msg, messages, attempts)
            if outcome.done:
                last_text = outcome.final_text
                break
        else:
            error_msg = f"Reached max_turns={self.max_turns} without producing a final answer."

        if store is not None:
            # Append only the in-flight tail back to the store (mutated in place,
            # returned to the caller, but not persisted here).
            store.extend(messages[prior_len:])

        elapsed = int((time.monotonic() - t0) * 1000)
        await _emit(
            AgentEvent(
                "agent_end",
                {"turns": attempts, "elapsed_ms": elapsed, "error": error_msg, "summary_len": len(last_text)},
            )
        )
        return _RunOutcome(attempts=attempts, last_text=last_text, error_msg=error_msg, memory=store)

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
