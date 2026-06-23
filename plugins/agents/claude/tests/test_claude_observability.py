"""Tests for Claude hook-based observability — tool outcomes rendered into the report.

The hooks observe only (each returns an empty decision) and feed a timeline; tests use
a mock timeline and invoke the registered callbacks directly (no real CLI/model).
"""

from unittest.mock import MagicMock

import pytest
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

from flyteplugins.agents.claude._run import _install_tool_hooks, _stringify


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
