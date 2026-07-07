"""Shared report timeline — render agent events into the Flyte task report.

Each adapter maps its own SDK's trace/span/event shape onto :class:`ReportTimeline`
rows; the row formatting, abbreviation, and best-effort error handling now live in
``flyte.report`` (:class:`flyte.report.Timeline`), so the native ``flyte.ai.agents``
loop and every adapter render through one implementation. This module keeps the
adapter-facing names (``ReportTimeline`` defaulting to the ``Agent`` tab, plus the
``abbrev``/``duration_ms``/``flush_report`` helpers) stable.
"""

from __future__ import annotations

import flyte.report
from flyte.report import Timeline, duration_ms
from flyte.report import abbreviate as abbrev

__all__ = ["ReportTimeline", "abbrev", "duration_ms", "flush_report"]


class ReportTimeline(Timeline):
    """A :class:`flyte.report.Timeline` that defaults to the ``Agent`` report tab."""

    def __init__(self, tab_name: str = "Agent"):
        super().__init__(tab_name)


async def flush_report() -> None:
    """Flush the active Flyte report — a best-effort no-op when there is none.

    Adapters call this once after a run so the rendered timeline is published.
    """
    try:
        await flyte.report.flush()
    except Exception:  # pragma: no cover - no active report / local run
        pass
