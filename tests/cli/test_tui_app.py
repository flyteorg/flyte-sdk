"""Tests for shared Flyte TUI widgets."""

from __future__ import annotations

import asyncio

from textual.app import App
from textual.widgets import Input, TextArea

from flyte.cli._tui._app import ConditionInputPanel, _condition_prompt_for_tui, _normalize_markdown_prompt_for_tui
from flyte.cli._tui._tracker import PendingCondition


def test_normalize_markdown_prompt_strips_html_break_before_list():
    prompt = (
        "#### Release tag\n\n"
        "All checks passed &#127881;. Enter the tag to deploy, e.g. "
        "<code>v2.1.0</code>.<br>\n"
        "- Use semver: `MAJOR.MINOR.PATCH`\n"
        "- Must be unique per release"
    )
    normalized = _normalize_markdown_prompt_for_tui(prompt)
    assert "<br>" not in normalized
    assert "<code>" not in normalized
    assert "🎉" in normalized
    assert "`v2.1.0`" in normalized
    assert "- Use semver: `MAJOR.MINOR.PATCH`" in normalized
    assert "- Must be unique per release" in normalized


def test_condition_prompt_for_tui_uses_compact_text_for_bullet_lists():
    prompt = "#### Release tag\n\nEnter a tag.\n\n- Use semver: `MAJOR.MINOR.PATCH`\n- Must be unique per release"
    mode, text = _condition_prompt_for_tui(prompt, "markdown")
    assert mode == "text"
    assert "Release tag" in text
    assert "####" not in text
    assert "\n\n- Use semver" not in text


def test_condition_prompt_for_tui_keeps_markdown_for_tables():
    prompt = "## Review needed\n\n| Metric | Value |\n|--------|-------|\n| Accuracy | 0.95 |\n"
    mode, text = _condition_prompt_for_tui(prompt, "markdown")
    assert mode == "markdown"
    assert "| Accuracy |" in text


def test_condition_input_panel_uses_text_area_for_str():
    async def run() -> None:
        pending = PendingCondition(
            action_id="a1",
            condition_name="notes",
            prompt="Enter notes",
            prompt_type="text",
            data_type=str,
        )

        class T(App):
            def compose(self):
                yield ConditionInputPanel(pending)

        app = T()
        async with app.run_test(size=(60, 20)) as pilot:
            await pilot.pause()
            widget = app.query_one("#condition-input")
            assert isinstance(widget, TextArea)
            widget.text = "line one\nline two"
            panel = app.query_one(ConditionInputPanel)
            assert panel._input_text() == "line one\nline two"

    asyncio.run(run())


def test_condition_input_panel_uses_input_for_int():
    async def run() -> None:
        pending = PendingCondition(
            action_id="a1",
            condition_name="count",
            prompt="Enter count",
            prompt_type="text",
            data_type=int,
        )

        class T(App):
            def compose(self):
                yield ConditionInputPanel(pending)

        app = T()
        async with app.run_test(size=(60, 12)) as pilot:
            await pilot.pause()
            widget = app.query_one("#condition-input")
            assert isinstance(widget, Input)

    asyncio.run(run())
