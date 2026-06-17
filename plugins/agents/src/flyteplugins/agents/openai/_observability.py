"""Forward the OpenAI Agents trace into the Flyte task report.

The OpenAI Agents SDK emits a structured trace of every run (agent spans, model
``generation``/``response`` turns, ``function`` tool calls, ``handoff`` and
``guardrail`` spans). :class:`FlyteTracingProcessor` is a ``TracingProcessor``
that renders those spans, as they finish, into a tab of the enclosing task's
Flyte **report** — giving an in-run timeline with timings, token usage and tool
inputs/outputs, alongside the tool tasks that already show up as Flyte actions.

It is best-effort by contract: a processor must never raise into the agent loop,
and report writes are skipped silently when there is no active report (e.g. the
task was not created with ``report=True``, or you are running locally).
"""

from __future__ import annotations

import html
import typing
from datetime import datetime

import flyte.report
from flyte._logging import logger

from agents import add_trace_processor, set_trace_processors
from agents.tracing.processor_interface import TracingProcessor

_ICONS = {
    "agent": "🤖",
    "generation": "🧠",
    "response": "🧠",
    "function": "🛠️",
    "handoff": "🔀",
    "guardrail": "🛡️",
    "mcp_tools": "🔌",
}


def _duration_ms(span: typing.Any) -> str:
    started, ended = getattr(span, "started_at", None), getattr(span, "ended_at", None)
    if not started or not ended:
        return ""
    try:
        delta = datetime.fromisoformat(ended.replace("Z", "+00:00")) - datetime.fromisoformat(
            started.replace("Z", "+00:00")
        )
        return f"{delta.total_seconds() * 1000:.0f} ms"
    except Exception:
        return ""


def _abbrev(value: typing.Any, limit: int = 300) -> str:
    text = "" if value is None else str(value)
    text = text if len(text) <= limit else text[:limit] + f"… (+{len(text) - limit})"
    return html.escape(text)


def _summarize(kind: str, export: dict[str, typing.Any]) -> str:
    if kind == "function":
        return f"<code>{_abbrev(export.get('input'), 160)}</code> → <code>{_abbrev(export.get('output'), 160)}</code>"
    if kind in ("generation", "response"):
        usage = export.get("usage") or {}
        if usage:
            parts = ", ".join(f"{k}={v}" for k, v in usage.items())
            return f'<span style="opacity:.7">{html.escape(parts)}</span>'
        return ""
    if kind == "handoff":
        return f"{_abbrev(export.get('from_agent'))} → {_abbrev(export.get('to_agent'))}"
    return ""


class FlyteTracingProcessor(TracingProcessor):
    """Render OpenAI Agents spans into a tab of the Flyte task report."""

    def __init__(self, tab_name: str = "Agent"):
        self._tab_name = tab_name

    def on_trace_start(self, trace: typing.Any) -> None:
        name = getattr(trace, "name", None) or "agent run"
        self._log(f'<h3 style="margin:8px 0 4px">{html.escape(str(name))}</h3>')

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
        label = export.get("name") or kind
        icon = _ICONS.get(kind, "•")
        duration = _duration_ms(span)
        detail = _summarize(kind, export)
        error = getattr(span, "error", None)
        error_html = f' <span style="color:#d33">⚠ {_abbrev(error, 200)}</span>' if error else ""
        meta = " · ".join(p for p in (kind, duration) if p)
        self._log(
            '<div style="padding:3px 0;border-bottom:1px solid rgba(128,128,128,.18);font-size:13px">'
            f"{icon} <b>{html.escape(str(label))}</b> "
            f'<span style="opacity:.55">{html.escape(meta)}</span>'
            f"{(' — ' + detail) if detail else ''}{error_html}"
            "</div>"
        )

    def _log(self, content_html: str) -> None:
        try:
            flyte.report.get_tab(self._tab_name).log(content_html)
        except Exception:
            pass  # no active report — nothing to render into


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
