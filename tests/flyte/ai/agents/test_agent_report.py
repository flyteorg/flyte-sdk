"""Tests for rendering the native agent loop's progress into a Flyte report."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flyte.ai.agents import Agent, LLMMessage, agent_progress_cb
from flyte.ai.agents._report import build_report_callback, render_event
from flyte.ai.agents.agent import AgentEvent


def _row_kwargs(timeline: MagicMock):
    return {c.kwargs.get("meta", ""): c.kwargs for c in timeline.row.call_args_list}


def test_render_event_maps_the_event_kinds_to_rows():
    tl = MagicMock()
    render_event(tl, AgentEvent("agent_start", {"name": "bot", "model": "m"}))
    render_event(tl, AgentEvent("tool_start", {"tool": "search", "args": {"q": "x"}}))
    render_event(tl, AgentEvent("tool_end", {"tool": "search", "result": "hits"}))
    render_event(tl, AgentEvent("tool_error", {"tool": "search", "error": "boom"}))
    render_event(tl, AgentEvent("agent_end", {"turns": 3, "elapsed_ms": 12, "error": ""}))

    tl.heading.assert_called_once()  # agent_start -> heading
    rows = _row_kwargs(tl)
    assert rows["tool"]["label"] == "search"
    assert rows["result"]["label"] == "search"
    assert rows["error"]["error"] == "boom"
    # agent_end row carries the turn/elapsed summary and no error.
    end = next(c.kwargs for c in tl.row.call_args_list if c.kwargs.get("label") == "done")
    assert "3 turns" in end["meta"] and "12 ms" in end["meta"]
    assert end["error"] is None


def test_render_event_skips_empty_assistant_messages():
    tl = MagicMock()
    render_event(tl, AgentEvent("message", {"role": "assistant", "content": ""}))
    tl.row.assert_not_called()
    render_event(tl, AgentEvent("message", {"role": "assistant", "content": "hello"}))
    tl.row.assert_called_once()


@pytest.mark.asyncio
async def test_build_report_callback_chains_inner_then_renders():
    inner = AsyncMock()
    with patch("flyte.ai.agents._report.render_event") as render:
        cb = build_report_callback("Agent", inner)
        event = AgentEvent("tool_start", {"tool": "t"})
        await cb(event)

    inner.assert_awaited_once_with(event)
    render.assert_called_once()
    assert render.call_args.args[1] is event


@pytest.mark.asyncio
async def test_build_report_callback_survives_a_render_error():
    inner = AsyncMock()
    with patch("flyte.ai.agents._report.render_event", side_effect=RuntimeError("bad")):
        cb = build_report_callback("Agent", inner)
        await cb(AgentEvent("agent_start", {}))  # must not raise
    inner.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_run_renders_a_report_timeline():
    llm = AsyncMock(return_value=LLMMessage(content="hello", tool_calls=[]))
    agent = Agent(name="t", model="m", call_llm=llm)

    tab = MagicMock()
    with patch("flyte.report.get_tab", return_value=tab):
        result = await agent.run.aio("hi", [])

    assert result.summary == "hello"
    logged = " ".join(call.args[0] for call in tab.log.call_args_list)
    # The heading (agent_start), the assistant message, and the done row all landed.
    assert "t · m" in logged
    assert "hello" in logged
    assert "done" in logged


@pytest.mark.asyncio
async def test_agent_run_preserves_a_caller_installed_progress_cb():
    seen = []

    async def user_cb(event):
        seen.append(event.type)

    token = agent_progress_cb.set(user_cb)
    try:
        llm = AsyncMock(return_value=LLMMessage(content="done", tool_calls=[]))
        agent = Agent(name="t", model="m", call_llm=llm)
        await agent.run.aio("hi", [])
    finally:
        agent_progress_cb.reset(token)

    # The user's callback still received the loop's events (report rendering is additive).
    assert "agent_start" in seen and "agent_end" in seen
