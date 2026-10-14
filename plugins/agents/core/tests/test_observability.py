"""Tests for the shared observability primitives every adapter renders through.

The report timeline and its helpers live in core so all adapters render consistently;
these cover the formatting + the best-effort "never break the agent loop" contract.
"""

from unittest.mock import MagicMock, patch

import pytest

from flyteplugins.agents.core import ReportTimeline, abbrev, duration_ms, flush_report


def test_abbrev_collapses_long_text_into_an_expandable_details():
    out = abbrev("x" * 350, limit=300)
    # The preview marks the overflow, and the full content is retained inside an
    # expandable <details> instead of being truncated away (so the user can open it).
    assert "(+50)" in out
    assert "<details" in out and "</details>" in out
    assert "x" * 350 in out


def test_abbrev_caps_the_expanded_body_of_enormous_values():
    out = abbrev("y" * 60000, limit=300)
    # Expandable, but the body is capped so one giant result can't bloat the report.
    assert "<details" in out
    assert "truncated" in out
    assert out.count("y") < 60000


def test_abbrev_handles_none_and_short_values():
    assert abbrev(None) == ""
    assert abbrev("hi") == "hi"
    assert abbrev(42) == "42"


def test_abbrev_html_escapes():
    assert "&lt;script&gt;" in abbrev("<script>")


def test_duration_ms_formats_the_iso_gap():
    assert duration_ms("2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z") == "1000 ms"


def test_duration_ms_empty_on_missing_or_unparseable_input():
    assert duration_ms("", "2024-01-01T00:00:01Z") == ""
    assert duration_ms(None, None) == ""
    assert duration_ms("not-a-date", "also-not") == ""


def test_report_timeline_is_silent_without_an_active_report():
    # Local / unit-test context: there is no report, so writes must never raise.
    tl = ReportTimeline()
    tl.heading("Agent")
    tl.row(icon="🛠️", label="get_weather", meta="tool", detail="Paris")
    tl.row(label="boom", error="it failed")


def test_report_timeline_logs_heading_and_rows_into_the_tab():
    tab = MagicMock()
    with patch("flyte.report.get_tab", return_value=tab):
        tl = ReportTimeline("Agent")
        tl.heading("Hi")
        tl.row(icon="💬", label="assistant", detail="hello")

    assert tab.log.call_count == 2  # heading + row
    logged = " ".join(call.args[0] for call in tab.log.call_args_list)
    assert "assistant" in logged and "hello" in logged


@pytest.mark.asyncio
async def test_flush_report_is_a_noop_without_an_active_report():
    await flush_report()  # must not raise outside a task/report context
