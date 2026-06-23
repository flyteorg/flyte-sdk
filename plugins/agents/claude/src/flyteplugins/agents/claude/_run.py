"""``run_agent`` — run a Claude Agent SDK agent on Flyte.

The Claude Agent SDK owns the loop (it drives the model via the Claude Code
runtime). ``run_agent`` runs that loop inside your ``@env.task``: it builds an
in-process MCP server from the tools, points the SDK at it, streams the run, and
renders the timeline into the Flyte report.

Durability: tool calls are durable Flyte child actions (see
:func:`flyteplugins.agents.claude.function_tool`). Per-turn model replay is not
available here — the model loop runs in the Claude Code runtime (a subprocess
Flyte doesn't intercept), so a model turn can't be a ``flyte.trace`` leaf the way
it is for client-side SDKs. Instead, ``durable=True`` wires the SDK's own session
mirror + resume onto a :class:`flyte.Checkpoint` (see :mod:`._durable`), so a
crashed attempt's conversation is restored on retry rather than restarted. Tool
durability, retries and caching apply regardless.

Observability: beyond streaming the assistant turns, ``PostToolUse`` /
``PostToolUseFailure`` hooks record each tool's outcome (result or error) into
the report timeline.
"""

from __future__ import annotations

import json
import typing

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    HookMatcher,
    ResultMessage,
    SdkMcpTool,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    query,
)
from flyte._task import TaskTemplate
from flyteplugins.agents.core import ReportTimeline, abbrev, flush_report

from ._durable import wire_durable_session
from ._memory import wire_memory_session
from ._tools import function_tool


def _coerce_tool(t: typing.Any) -> SdkMcpTool:
    if isinstance(t, SdkMcpTool):
        return t
    if isinstance(t, TaskTemplate):
        return function_tool(t)
    return t


async def run_agent(
    input: str,
    *,
    tools: typing.Sequence[typing.Any] = (),
    model: str | None = "claude-sonnet-4-5",
    instructions: str | None = None,
    max_turns: int | None = None,
    durable: bool = True,
    observability: bool = True,
    options: ClaudeAgentOptions | None = None,
    server_name: str = "flyte_tools",
    memory_key: str | None = None,
) -> str:
    """Run a Claude agent with the given tools and prompt; return the final text.

    Call this from inside an ``@env.task`` — that task is the durable parent,
    and each tool the agent calls runs as a durable Flyte child action. Pass a
    fully-built ``ClaudeAgentOptions`` via ``options`` to keep SDK-native config
    (subagents, permissions, hooks, session resume); ``tools``/``model``/
    ``instructions``/``max_turns`` are layered on top.

    With ``durable=True`` (and a checkpoint-capable task context) the SDK's session
    mirror + resume is wired onto a ``flyte.Checkpoint``, so a retry resumes the
    conversation instead of restarting it. With ``observability=True`` the run
    timeline — assistant turns plus per-tool outcomes (via hooks) — is rendered into
    the task report.

    Set ``memory_key`` (a user/thread id) for **cross-run memory**: the transcript is
    persisted to a durable, keyed ``MemoryStore`` and resumed on a later run with the
    same key (this also covers crash-resume, so it takes precedence over the per-run
    ``durable`` checkpoint).

    The ``claude-agent-sdk`` wheel bundles the native ``claude`` CLI, so the runtime
    image needs no separate Node.js install — just an Anthropic API key.
    """
    sdk_tools = [_coerce_tool(t) for t in tools]
    opts = options or ClaudeAgentOptions()

    if sdk_tools:
        server = create_sdk_mcp_server(server_name, tools=sdk_tools)
        opts.mcp_servers = {**(opts.mcp_servers or {}), server_name: server}
        allowed = [f"mcp__{server_name}__{t.name}" for t in sdk_tools]
        opts.allowed_tools = [*(opts.allowed_tools or []), *allowed]
    if instructions is not None:
        opts.system_prompt = instructions
    if model is not None:
        opts.model = model
    if max_turns is not None:
        opts.max_turns = max_turns

    if memory_key:
        await wire_memory_session(opts, memory_key=memory_key)
    else:
        await wire_durable_session(opts, durable=durable)

    timeline = ReportTimeline() if observability else None
    if timeline is not None:
        timeline.heading("Claude agent")
        _install_tool_hooks(opts, timeline)

    final = ""
    async for message in query(prompt=input, options=opts):
        if isinstance(message, AssistantMessage):
            if timeline is not None:
                _render_assistant(timeline, message)
        elif isinstance(message, ResultMessage):
            final = message.result or final
            if timeline is not None:
                _render_result(timeline, message)

    if observability:
        await flush_report()
    return final or ""


def _render_assistant(timeline: ReportTimeline, message: AssistantMessage) -> None:
    for block in message.content:
        if isinstance(block, TextBlock):
            if block.text.strip():
                timeline.row(icon="💬", label="assistant", detail=abbrev(block.text, 200))
        elif isinstance(block, ToolUseBlock):
            timeline.row(icon="🛠️", label=block.name, meta="tool", detail=abbrev(block.input, 160))


def _render_result(timeline: ReportTimeline, message: ResultMessage) -> None:
    parts = []
    if message.num_turns:
        parts.append(f"{message.num_turns} turns")
    if message.duration_ms:
        parts.append(f"{message.duration_ms} ms")
    if message.total_cost_usd:
        parts.append(f"${message.total_cost_usd:.4f}")
    timeline.row(icon="✅", label="result", meta=" · ".join(parts), error="error" if message.is_error else None)


def _stringify(value: typing.Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _install_tool_hooks(opts: ClaudeAgentOptions, timeline: ReportTimeline) -> None:
    """Record each tool's *outcome* into the report via PostToolUse hooks.

    The streamed assistant turns already show the tool *request* (``ToolUseBlock``);
    these hooks add the result/error the message stream doesn't surface. They observe
    only — each returns an empty decision — and are merged into any user-provided
    ``opts.hooks`` rather than replacing them.
    """

    async def _post_tool(input_data: dict, tool_use_id: str | None, context: typing.Any) -> dict:
        timeline.row(
            icon="🔧",
            label=input_data.get("tool_name", ""),
            meta="tool result",
            detail=abbrev(_stringify(input_data.get("tool_response", "")), 160),
        )
        return {}

    async def _post_tool_failure(input_data: dict, tool_use_id: str | None, context: typing.Any) -> dict:
        timeline.row(
            icon="❌",
            label=input_data.get("tool_name", ""),
            meta="tool error",
            detail=abbrev(_stringify(input_data.get("error", "")), 160),
            error="error",
        )
        return {}

    additions = {
        "PostToolUse": [HookMatcher(hooks=[_post_tool])],
        "PostToolUseFailure": [HookMatcher(hooks=[_post_tool_failure])],
    }
    merged = dict(opts.hooks or {})
    for event, matchers in additions.items():
        merged[event] = [*(merged.get(event) or []), *matchers]
    opts.hooks = merged
