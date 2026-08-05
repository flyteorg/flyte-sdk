"""Render the agent loop's progress events into a Flyte report tab.

The loop already emits normalized :class:`~flyte.ai.agents.agent.AgentEvent`s through
:data:`~flyte.ai.agents.agent.agent_progress_cb`; this module turns those into rows on a
:class:`flyte.report.Timeline` — the same widget the ``flyteplugins.agents.*`` adapters
render through, so a native agent and an adapter-wrapped agent produce a consistent
report. It is a plain subscriber: it chains onto any callback the caller already
installed, and it is best-effort (a no-op when there is no active report, and it never
lets a render error escape into the loop).
"""

from __future__ import annotations

import typing

from flyte._logging import logger
from flyte.report import Timeline, abbreviate

if typing.TYPE_CHECKING:
    from .agent import AgentEvent, AgentProgressCallback


def render_event(timeline: Timeline, event: "AgentEvent") -> None:
    """Map one :class:`AgentEvent` onto a heading or row of ``timeline``."""
    data = event.data
    kind = event.type
    if kind == "agent_start":
        label = data.get("name") or "agent"
        model = data.get("model") or ""
        mode = f" · {data['mode']}" if data.get("mode") else ""
        timeline.heading(f"{label} · {model}{mode}" if model else f"{label}{mode}")
    elif kind == "tool_start":
        timeline.row(
            icon="🛠️",
            label=data.get("tool", ""),
            meta="tool",
            detail=abbreviate(data.get("args") if "args" in data else data.get("code"), 200),
        )
    elif kind == "tool_end":
        timeline.row(icon="✅", label=data.get("tool", ""), meta="result", detail=abbreviate(data.get("result")))
    elif kind == "tool_error":
        timeline.row(icon="⚠️", label=data.get("tool", ""), meta="error", error=data.get("error"))
    elif kind == "approval_request":
        timeline.row(
            icon="⏸️", label=data.get("tool", ""), meta="awaiting approval", detail=abbreviate(data.get("args"), 200)
        )
    elif kind == "approval_decision":
        approved = data.get("approved")
        timeline.row(
            icon="✅" if approved else "🚫",
            label=data.get("tool", ""),
            meta="approved" if approved else "rejected",
        )
    elif kind == "message":
        content = data.get("content")
        if content:
            timeline.row(icon="💬", label=data.get("role", "assistant"), detail=abbreviate(content))
    elif kind == "agent_end":
        meta = f"{data.get('turns', '?')} turns"
        if data.get("elapsed_ms") is not None:
            meta += f" · {data['elapsed_ms']} ms"
        timeline.row(icon="🏁", label="done", meta=meta, error=data.get("error") or None)


def build_report_callback(tab: str, inner: "AgentProgressCallback | None") -> "AgentProgressCallback":
    """A progress callback that renders each event into ``tab`` after ``inner`` runs.

    ``inner`` (any callback already installed by the caller) is awaited first, so
    report rendering is additive — it never displaces a user's own subscriber. A render
    failure is swallowed so observability cannot break the loop.
    """
    timeline = Timeline(tab)

    async def _cb(event: "AgentEvent") -> None:
        if inner is not None:
            await inner(event)
        try:
            render_event(timeline, event)
        except Exception:  # pragma: no cover - rendering must never break the loop
            logger.debug("Agent report renderer raised; suppressing", exc_info=True)

    return _cb
