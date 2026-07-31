"""Tests for the reusable report widgets (abbreviate / duration_ms / Timeline)."""

from unittest.mock import MagicMock, patch

from flyte.report import Timeline, abbreviate, duration_ms


def test_abbreviate_collapses_long_text_into_an_expandable_details():
    out = abbreviate("x" * 350, limit=300)
    assert "(+50)" in out
    assert "<details" in out and "</details>" in out
    assert "x" * 350 in out


def test_abbreviate_caps_the_expanded_body_of_enormous_values():
    out = abbreviate("y" * 60000, limit=300)
    assert "<details" in out
    assert "truncated" in out
    assert out.count("y") < 60000


def test_abbreviate_handles_none_and_short_values():
    assert abbreviate(None) == ""
    assert abbreviate("hi") == "hi"
    assert abbreviate(42) == "42"


def test_abbreviate_html_escapes():
    assert "&lt;script&gt;" in abbreviate("<script>")


def test_duration_ms_formats_the_iso_gap():
    assert duration_ms("2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z") == "1000 ms"


def test_duration_ms_empty_on_missing_or_unparseable_input():
    assert duration_ms("", "2024-01-01T00:00:01Z") == ""
    assert duration_ms(None, None) == ""
    assert duration_ms("not-a-date", "also-not") == ""


def test_timeline_is_silent_without_an_active_report():
    # Local / unit-test context: there is no report, so writes must never raise.
    tl = Timeline()
    tl.heading("Run")
    tl.row(icon="🛠️", label="get_weather", meta="tool", detail="Paris")
    tl.row(label="boom", error="it failed")


def test_timeline_logs_heading_and_rows_into_the_named_tab():
    tab = MagicMock()
    with patch("flyte.report.get_tab", return_value=tab) as get_tab:
        tl = Timeline("Agent")
        tl.heading("Hi")
        tl.row(icon="💬", label="assistant", detail="hello")

    get_tab.assert_called_with("Agent")
    assert tab.log.call_count == 2  # heading + row
    logged = " ".join(call.args[0] for call in tab.log.call_args_list)
    assert "assistant" in logged and "hello" in logged
