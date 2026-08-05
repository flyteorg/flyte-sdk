"""Tests for Claude hook-based observability — tool outcomes rendered into the report.

The hooks observe only (each returns an empty decision) and feed a timeline; tests use
a mock timeline and invoke the registered callbacks directly (no real CLI/model).
"""

from unittest.mock import MagicMock

import pytest
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

from flyteplugins.agents.claude._run import _fmt_usage, _install_tool_hooks, _stringify


def _callback(opts, event):
    return opts.hooks[event][0].hooks[0]


def test_install_registers_post_tool_events():
    opts = ClaudeAgentOptions()
    _install_tool_hooks(opts, MagicMock())
    assert "PostToolUse" in opts.hooks
    assert "PostToolUseFailure" in opts.hooks
    assert isinstance(opts.hooks["PostToolUse"][0], HookMatcher)


@pytest.mark.asyncio
async def test_post_tool_hook_records_result_row():
    opts = ClaudeAgentOptions()
    timeline = MagicMock()
    _install_tool_hooks(opts, timeline)

    out = await _callback(opts, "PostToolUse")({"tool_name": "get_weather", "tool_response": {"t": 22}}, "tid", {})

    assert out == {}  # observe-only: no decision
    timeline.row.assert_called_once()
    kwargs = timeline.row.call_args.kwargs
    assert kwargs["label"] == "get_weather"
    assert kwargs["meta"] == "tool result"


@pytest.mark.asyncio
async def test_post_tool_failure_hook_records_error_row():
    opts = ClaudeAgentOptions()
    timeline = MagicMock()
    _install_tool_hooks(opts, timeline)

    out = await _callback(opts, "PostToolUseFailure")({"tool_name": "boom", "error": "kaboom"}, "tid", {})

    assert out == {}
    kwargs = timeline.row.call_args.kwargs
    assert kwargs["label"] == "boom"
    assert kwargs["meta"] == "tool error"
    assert kwargs["error"] == "error"


def test_install_merges_with_existing_user_hooks():
    user_matcher = HookMatcher(hooks=[])
    opts = ClaudeAgentOptions(hooks={"PostToolUse": [user_matcher]})
    _install_tool_hooks(opts, MagicMock())

    # the user's matcher is preserved and ours is appended (not replaced)
    assert opts.hooks["PostToolUse"][0] is user_matcher
    assert len(opts.hooks["PostToolUse"]) == 2


def test_stringify_handles_str_and_objects():
    assert _stringify("hi") == "hi"
    assert _stringify({"a": 1}) == '{"a": 1}'


def test_fmt_usage_breaks_down_tokens_that_drive_cost():
    out = _fmt_usage(
        {
            "input_tokens": 1200,
            "output_tokens": 340,
            "cache_read_input_tokens": 5000,
            "cache_creation_input_tokens": 1100,
        }
    )
    assert out == "in 1.2k · out 340 · cache read 5.0k · cache write 1.1k"


def test_fmt_usage_is_defensive_about_keys_and_empties():
    assert _fmt_usage(None) == ""
    assert _fmt_usage({}) == ""
    assert _fmt_usage({"output_tokens": 0}) == ""  # zero-valued fields are dropped
    assert _fmt_usage({"inputTokens": 500}) == "in 500"  # camelCase fallback
