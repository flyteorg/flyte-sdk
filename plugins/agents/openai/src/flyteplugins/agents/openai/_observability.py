"""Forward the OpenAI Agents trace into the Flyte task report.

The OpenAI Agents SDK emits a structured trace of every run (agent spans, model
``generation``/``response`` turns, ``function`` tool calls, ``handoff`` and
``guardrail`` spans). :class:`FlyteTracingProcessor` is a ``TracingProcessor``
that maps those spans onto the shared :class:`~flyteplugins.agents.core.ReportTimeline`,
rendering an in-run timeline (timings, token usage, tool inputs/outputs) into a tab of
the enclosing task's Flyte report, alongside the tool tasks that already show up as
Flyte actions.

It is best-effort by contract: a processor must never raise into the agent loop,
and report writes are skipped silently when there is no active report.
"""

from __future__ import annotations

import typing

from agents import add_trace_processor, set_trace_processors
from agents.tracing.processor_interface import TracingProcessor
from flyte._logging import logger
from flyteplugins.agents.core import ReportTimeline, abbrev, duration_ms

_ICONS = {
    "agent": "🤖",
    "generation": "🧠",
    "response": "🧠",
    "function": "🛠️",
    "handoff": "🔀",
    "guardrail": "🛡️",
    "mcp_tools": "🔌",
}


def _summarize(kind: str, export: dict[str, typing.Any]) -> str:
    """Render the OpenAI-specific span payload as a compact HTML detail string."""
    if kind == "function":
        return f"<code>{abbrev(export.get('input'), 160)}</code> → <code>{abbrev(export.get('output'), 160)}</code>"
    if kind in ("generation", "response"):
        usage = export.get("usage") or {}
        if usage:
            parts = ", ".join(f"{k}={v}" for k, v in usage.items())
            return f'<span style="opacity:.7">{abbrev(parts)}</span>'
        return ""
    if kind == "handoff":
        return f"{abbrev(export.get('from_agent'))} → {abbrev(export.get('to_agent'))}"
    return ""


class FlyteTracingProcessor(TracingProcessor):
    """Map OpenAI Agents spans onto the shared :class:`ReportTimeline`."""

    def __init__(self, tab_name: str = "Agent"):
        self._timeline = ReportTimeline(tab_name)

    def on_trace_start(self, trace: typing.Any) -> None:
        self._timeline.heading("OpenAI agent")

    def on_trace_end(self, trace: typing.Any) -> None:
        pass

    def on_span_start(self, span: typing.Any) -> None:
        pass

    def on_span_end(self, span: typing.Any) -> None:
        try:
            self._render(span)
        except Exception:  # pragma: no cover - observability must never break the loop
            logger.debug("FlyteTracingProcessor failed to render a span", exc_info=True)

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass

    def _render(self, span: typing.Any) -> None:
        data = getattr(span, "span_data", None)
        if data is None:
            return
        kind = getattr(data, "type", "custom")
        export = data.export() if hasattr(data, "export") else {}
        duration = duration_ms(getattr(span, "started_at", None), getattr(span, "ended_at", None))
        self._timeline.row(
            icon=_ICONS.get(kind, "•"),
            label=export.get("name") or kind,
            meta=" · ".join(p for p in (kind, duration) if p),
            detail=_summarize(kind, export),
            error=getattr(span, "error", None),
        )


def install_flyte_tracing(*, exclusive: bool = True, tab_name: str = "Agent") -> FlyteTracingProcessor:
    """Install a :class:`FlyteTracingProcessor` as a global trace processor.

    With ``exclusive=True`` (default) it replaces all processors, so traces are
    rendered only into the Flyte report and nothing is uploaded to OpenAI's
    tracing backend. Set ``exclusive=False`` to keep the SDK's default processors
    (e.g. to also export to the OpenAI dashboard) and add Flyte alongside.
    """
    processor = FlyteTracingProcessor(tab_name=tab_name)
    if exclusive:
        set_trace_processors([processor])
    else:
        add_trace_processor(processor)
    return processor
