"""`run_agent` — run a DeepSeek Harness agent on Flyte.

DeepSeek Harness owns the loop: its runtime subprocess drives the model and its
own tools (bash, string editor, ...) over JSON-RPC stdio. `run_agent` runs that
loop inside your `@env.task`: it prepares the workspace and session store,
publishes your Flyte-task tools into the workspace (see `._bridge`), drives
`DeepSeekHarness.run` — a blocking call, bridged off the event loop with
`asyncio.to_thread` so the bridge can serve tool calls while it runs — and
returns the final response.

Durability: tool calls are durable Flyte child actions (see
`flyteplugins.agents.deepseek.tool`). Per-turn model replay is not available
here — the model loop runs inside the harness runtime (a subprocess Flyte
doesn't intercept), so a model turn can't be a `flyte.trace` leaf the way it is
for client-side SDKs. Instead, `durable=True` wires the harness's own JSONL
session store onto a `flyte.Checkpoint` (see `._durable`), so a crashed
attempt's conversation is restored on retry rather than restarted. Tool
durability, retries and caching apply regardless.

Observability: the harness streams session events as JSON-RPC notifications;
those are rendered into the Flyte report timeline alongside the per-tool
outcomes recorded by the bridge.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import shutil
import tempfile
import typing

from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig, Notification, RunResult
from flyte._logging import logger
from flyte._task import TaskTemplate
from flyteplugins.agents.core import (
    ReportTimeline,
    abbrev,
    apply_call_wrapper,
    apply_instrumentation,
    flush_report,
    sync_variant,
)

from ._bridge import ToolBridge
from ._durable import wire_durable_session
from ._memory import wire_memory_session
from ._tools import HarnessTool, tool


def _coerce_tool(t: typing.Any) -> HarnessTool:
    if isinstance(t, HarnessTool):
        return t
    if isinstance(t, TaskTemplate):
        return tool(t)  # type: ignore[return-value]
    return tool(t)  # type: ignore[return-value]


def _harness_run(harness: DeepSeekHarness, prompt: str, **kwargs: typing.Any) -> RunResult:
    """The SDK call itself, isolated so an observability library can wrap it."""
    return harness.run(prompt, **kwargs)


async def run_agent(
    input: str,
    *,
    tools: typing.Sequence[typing.Any] = (),
    model: str | None = None,
    instructions: str | None = None,
    durable: bool = True,
    observability: bool = True,
    memory_key: str | None = None,
    provider: str | None = None,
    max_tokens: int | None = None,
    workspace: str | os.PathLike | None = None,
    session_id: str | None = None,
    config: DeepSeekHarnessConfig | None = None,
    **harness_kwargs: typing.Any,
) -> str:
    """Run a DeepSeek Harness agent with the given tools and prompt; return the final text.

    Await this from an async task as `await run_agent(...)`; from a sync task
    use `flyteplugins.agents.deepseek.run_agent_sync` instead.

    Call this from inside an `@env.task` — that task is the durable parent, and
    each tool the agent calls runs as a durable Flyte child action. Give the
    enclosing task `retries=...` for self-healing and `report=True` to see the
    agent timeline.

    Pass a fully-built `DeepSeekHarnessConfig` via `config` to keep SDK-native
    configuration (a custom `cordis` composition, `base_url` / `api_key`,
    timeouts); `model` / `provider` / `max_tokens` / `workspace` are layered on
    top of it, and any extra `**harness_kwargs` are applied last.

    Args:
        input: The user prompt.
        tools: `tool`-wrapped tools or bare `@env.task` templates. They are
            published into the harness workspace as executable shims and
            described to the model in its instructions, because the harness has
            no tool-registration channel of its own — see `._bridge`.
        model: Model id resolved by the configured provider. `None` leaves the
            SDK default (`deepseek-v4-flash`) in place.
        instructions: System prompt, passed to the runtime as `DSH_SYSTEM_PROMPT`.
        durable: Wire the harness's JSONL session store onto a `flyte.Checkpoint`
            so a retry resumes the conversation instead of restarting it.
        observability: Render the run timeline into the Flyte task report.
        memory_key: Stable id (a user/thread id) for cross-run memory. When set,
            the session is persisted to a keyed `MemoryStore` and resumed on a
            later run with the same key. This also covers crash-resume, so it
            takes precedence over the per-run `durable` checkpoint.
        provider: Provider route registered by the Cordis composition.
        max_tokens: Per-request output-token cap for the root agent.
        workspace: Directory the harness works in (its bash cwd). Defaults to a
            fresh temp directory removed when the run ends; pass a path — for
            example a `flyte.io.Dir` you downloaded — to work on real files.
        session_id: Override the session id. Defaults to the durable/memory
            session id, so retries and same-key runs continue one conversation.
        config: A pre-built `DeepSeekHarnessConfig` to layer the above onto.
        **harness_kwargs: Extra `DeepSeekHarnessConfig` fields (e.g. `cordis=`,
            `base_url=`, `api_key=`, `request_timeout_seconds=`).

    Returns:
        The agent's final response as a string.
    """
    harness_tools = [_coerce_tool(t) for t in tools]

    owned_workspace = workspace is None
    workspace_path = _prepare_workspace(workspace)
    session_root = pathlib.Path(tempfile.mkdtemp(prefix="flyte-dsh-sessions-"))

    # Memory subsumes crash-resume: a keyed store survives retries too, so when a
    # memory_key is given it replaces the per-run checkpoint session.
    session = await wire_memory_session(session_root, memory_key=memory_key)
    if session is None:
        session = await wire_durable_session(session_root, durable=durable)

    timeline = ReportTimeline() if observability else None
    if timeline is not None:
        timeline.heading("DeepSeek Harness agent")

    bridge = ToolBridge(harness_tools, timeline=timeline)
    try:
        await bridge.start(workspace_path)

        cfg = _build_config(
            config,
            model=model,
            provider=provider,
            max_tokens=max_tokens,
            instructions=instructions,
            cwd=str(workspace_path),
            session_root=str(session_root),
            extra=harness_kwargs,
        )
        # The SDK carries its configuration on this object, so that is what is offered
        # to any registered instrumentor. Unregistered, it comes back unchanged.
        cfg = apply_instrumentation("deepseek", cfg)

        prompt = _compose_prompt(input, bridge.instructions())
        run_session_id = session_id or (session.session_id if session is not None else None)

        result = await _drive(cfg, prompt, run_session_id, timeline)

        if timeline is not None:
            _render_result(timeline, result)
        if session is not None:
            await session.persist(session_root)
        return result.final_response or ""
    finally:
        await bridge.stop()
        shutil.rmtree(session_root, ignore_errors=True)
        if owned_workspace:
            shutil.rmtree(workspace_path, ignore_errors=True)
        elif bridge.tools_dir is not None:
            # A caller-owned workspace is left as we found it, minus our shims.
            shutil.rmtree(bridge.tools_dir, ignore_errors=True)
        if observability:
            await flush_report()


run_agent_sync = sync_variant(run_agent)


async def _drive(
    cfg: DeepSeekHarnessConfig,
    prompt: str,
    session_id: str | None,
    timeline: ReportTimeline | None,
) -> RunResult:
    """Run the (blocking) harness off the event loop, streaming events to the report.

    `DeepSeekHarness.run` blocks until the agent goes idle, so it runs on a worker
    thread — that keeps the loop free to serve the tool bridge, which is what makes
    tool calls possible at all. Notifications arrive on that worker thread, so each
    is marshalled back onto the loop before touching the report.
    """
    loop = asyncio.get_running_loop()
    on_notification = _notification_sink(loop, timeline) if timeline is not None else None

    def _blocking() -> RunResult:
        with DeepSeekHarness(cfg) as harness:
            run = apply_call_wrapper("deepseek", _harness_run)
            kwargs: dict[str, typing.Any] = {"on_notification": on_notification}
            if session_id is not None:
                kwargs["session_id"] = session_id
            return run(harness, prompt, **kwargs)

    return await asyncio.to_thread(_blocking)


def _prepare_workspace(workspace: str | os.PathLike | None) -> pathlib.Path:
    """Resolve (or create) the directory the harness works in.

    Kept sync — local filesystem setup, done once before the run starts.
    """
    path = pathlib.Path(workspace).resolve() if workspace else pathlib.Path(tempfile.mkdtemp(prefix="flyte-dsh-ws-"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_config(
    base: DeepSeekHarnessConfig | None,
    *,
    model: str | None,
    provider: str | None,
    max_tokens: int | None,
    instructions: str | None,
    cwd: str,
    session_root: str,
    extra: dict[str, typing.Any],
) -> DeepSeekHarnessConfig:
    """Layer the adapter's options onto a (possibly caller-supplied) SDK config.

    The adapter owns `cwd` and `session_root`: the workspace is where the tool
    shims are published, and the session root is what the durable/memory session
    mirrors. Everything else is the caller's.
    """
    fields = {f: getattr(base, f) for f in DeepSeekHarnessConfig.__dataclass_fields__} if base else {}
    fields.update(extra)
    if model is not None:
        fields["model"] = model
    if provider is not None:
        fields["provider"] = provider
    if max_tokens is not None:
        fields["max_tokens"] = max_tokens
    fields["cwd"] = cwd
    fields["session_root"] = session_root
    if instructions is not None:
        fields["env"] = {**(fields.get("env") or {}), "DSH_SYSTEM_PROMPT": instructions}
    return DeepSeekHarnessConfig(**fields)


def _compose_prompt(input: str, tool_manual: str) -> str:
    """Prefix the tool manual to the prompt, so the model learns what it can call."""
    return f"{tool_manual}\n\n---\n\n{input}" if tool_manual else input


def _notification_sink(
    loop: asyncio.AbstractEventLoop,
    timeline: ReportTimeline,
) -> typing.Callable[[Notification], None]:
    """Build the `on_notification` callback that renders session events to the report."""

    def sink(notification: Notification) -> None:
        if notification.method != "session.event":
            return
        event = notification.payload.get("event")
        if isinstance(event, dict):
            loop.call_soon_threadsafe(_render_event, timeline, event)

    return sink


def _render_event(timeline: ReportTimeline, event: dict[str, typing.Any]) -> None:
    """Render one harness session event as a timeline row (best-effort).

    Maps the runtime's `SessionEventMap` types onto timeline rows. The harness's
    own tools (bash, editor) show up as `tool/call` + `tool/result`; Flyte-task
    tools are recorded separately by the bridge, which sees their arguments and
    results directly.
    """
    try:
        kind = str(event.get("type") or "")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if kind == "assistant/message":
            text = _assistant_text(data)
            if text.strip():
                timeline.row(icon="💬", label="assistant", meta=_fmt_usage(data.get("usage")), detail=abbrev(text, 200))
        elif kind == "tool/call":
            timeline.row(
                icon="🧰",
                label=str(data.get("name") or "tool"),
                meta="harness tool",
                detail=abbrev(data.get("arguments"), 160),
            )
        elif kind == "tool/result":
            error = data.get("error") if isinstance(data.get("error"), dict) else None
            timeline.row(
                icon="❌" if error else "🔧",
                label=str((error or {}).get("name") or "tool"),
                meta="harness tool result",
                detail=abbrev(data.get("message"), 160),
                error="error" if error else None,
            )
        elif kind == "turn/end":
            reason = data.get("reason") if isinstance(data.get("reason"), dict) else {}
            timeline.row(icon="↩️", label="turn", meta=str(reason.get("kind") or ""))
    except Exception:  # pragma: no cover - observability must never break the run
        logger.debug("Could not render a DeepSeek harness event", exc_info=True)


def _compact(n: int) -> str:
    """Compact a token count, e.g. 5000 -> '5.0k'."""
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _fmt_usage(usage: typing.Any) -> str:
    """A compact token breakdown from an assistant turn's `TokenUsage`, if present."""
    if not isinstance(usage, dict):
        return ""

    def pick(*keys: str) -> int:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)) and value:
                return int(value)
        return 0

    fields = [
        ("in", pick("input_tokens", "inputTokens", "promptTokens")),
        ("out", pick("output_tokens", "outputTokens", "completionTokens")),
        ("cache read", pick("cache_read_input_tokens", "cacheReadInputTokens")),
    ]
    return " · ".join(f"{label} {_compact(n)}" for label, n in fields if n)


def _assistant_text(data: dict[str, typing.Any]) -> str:
    """Concatenate the text blocks of an `assistant/message` event."""
    message = data.get("message")
    owner = message if isinstance(message, dict) else data
    content = owner.get("content")
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get("text") or "") for block in content if isinstance(block, dict) and block.get("type") == "text"
    )


def _render_result(timeline: ReportTimeline, result: RunResult) -> None:
    parts = [f"{len(result.events)} events"]
    if result.finish_reason:
        parts.append(result.finish_reason)
    timeline.row(
        icon="✅",
        label="result",
        meta=" · ".join(parts),
        detail=abbrev(result.final_response, 200),
        error="error" if result.finish_reason == "error" else None,
    )
