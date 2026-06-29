"""Shared report timeline — render agent events into the Flyte task report.

Each adapter maps its own SDK's trace/span/event shape onto :class:`ReportTimeline`
rows; the report writing, row formatting, and best-effort error handling live
here so every adapter renders consistently.
"""

from __future__ import annotations

import html
import typing
from datetime import datetime

import flyte.report
from flyte._logging import logger

# Hard ceiling on the full body of an expanded row, so a single huge tool result
# (a big file read, a verbose API response) can't bloat the report unbounded.
_MAX_FULL_CHARS = 20000


def abbrev(value: typing.Any, limit: int = 300) -> str:
    """HTML-escape ``value`` for a report row.

    Short values render inline. Longer ones collapse into an expandable ``<details>``:
    the row shows a ``limit``-character preview with a ``+N`` overflow marker, and
    clicking it reveals the full content (up to a hard cap). Nothing is dropped on the
    floor, so a tool result that trails off in the report can always be opened in place.
    """
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return html.escape(text)
    preview = html.escape(text[:limit])
    body = html.escape(text[:_MAX_FULL_CHARS])
    if len(text) > _MAX_FULL_CHARS:
        body += f"\n... (+{len(text) - _MAX_FULL_CHARS} more, truncated)"
    return (
        '<details style="display:inline-block;vertical-align:top;max-width:100%">'
        f'<summary style="cursor:pointer">{preview}... (+{len(text) - limit})</summary>'
        f'<pre style="white-space:pre-wrap;word-break:break-word;margin:4px 0 0">{body}</pre>'
        "</details>"
    )


def duration_ms(start_iso: typing.Any, end_iso: typing.Any) -> str:
    """Format the gap between two ISO-8601 timestamps as ``"<n> ms"`` (best-effort)."""
    if not start_iso or not end_iso:
        return ""
    try:
        delta = datetime.fromisoformat(str(end_iso).replace("Z", "+00:00")) - datetime.fromisoformat(
            str(start_iso).replace("Z", "+00:00")
        )
        return f"{delta.total_seconds() * 1000:.0f} ms"
    except Exception:
        return ""


class ReportTimeline:
    """Append a best-effort timeline of agent events to a tab of the task report.

    Writes are skipped silently when there is no active report (e.g. the task was
    not created with ``report=True`` or you are running locally) — observability
    must never break the agent loop.
    """

    def __init__(self, tab_name: str = "Agent"):
        self._tab_name = tab_name

    def heading(self, text: typing.Any) -> None:
        self._log(f'<h3 style="margin:8px 0 4px">{html.escape(str(text))}</h3>')

    def row(
        self,
        *,
        icon: str = "•",
        label: typing.Any = "",
        meta: str = "",
        detail: str = "",
        error: typing.Any = None,
    ) -> None:
        error_html = f' <span style="color:#d33">⚠ {abbrev(error, 200)}</span>' if error else ""
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
        except Exception:  # pragma: no cover - no active report
            logger.debug("ReportTimeline: no active report to render into", exc_info=True)


async def flush_report() -> None:
    """Flush the active Flyte report — a best-effort no-op when there is none.

    Adapters call this once after a run so the rendered timeline is published.
    """
    try:
        await flyte.report.flush()
    except Exception:  # pragma: no cover - no active report / local run
        pass
