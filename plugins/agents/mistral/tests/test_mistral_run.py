"""Tests for the Mistral run_agent (SDK runner) + the durable-turn seam.

The agent loop itself is the SDK's (``conversations.run_async``), so these mock
it and verify our orchestration: tools are registered, the final text is
extracted, and the durable-turn wrapping records/replays a turn faithfully.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import flyte
import pytest

from flyteplugins.agents.mistral import run_agent, tool
from flyteplugins.agents.mistral._run import _final_text, _install_turn_hooks, _render, _UsageSink


def _message(text):
    e = MagicMock()
    e.type = "message.output"
    e.content = text
    return e


def test_render_emits_rows_for_tool_calls_and_assistant_messages():
    timeline = MagicMock()
    call = MagicMock(type="function.call", arguments='{"city": "Paris"}')
    call.name = "search_web"
    _render(timeline, [call, _message("the answer")])

    labels = [c.kwargs.get("label") for c in timeline.row.call_args_list]
    assert "search_web" in labels  # tool call rendered
    assert "assistant" in labels  # message.output rendered


@pytest.mark.asyncio
async def test_run_agent_registers_tools_and_returns_final_text(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    env = flyte.TaskEnvironment("mistral_run_a")

    @tool
    @env.task
    def get_weather(city: str) -> str:
        """Get weather."""
        return f"sunny in {city}"

    result = MagicMock()
    result.output_entries = [_message("It's sunny in SF!")]
    client = MagicMock()
    client.beta.conversations.run_async = AsyncMock(return_value=result)

    with patch("mistralai.client.Mistral", return_value=client):
        out = await run_agent("Weather in SF?", tools=[get_weather], durable=False, observability=False)

    assert out == "It's sunny in SF!"
    # The SDK runner was invoked once with our RunContext.
    assert client.beta.conversations.run_async.await_count == 1


@pytest.mark.asyncio
async def test_timeout_ms_threads_into_run_async(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    result = MagicMock()
    result.output_entries = [_message("done")]
    client = MagicMock()
    client.beta.conversations.run_async = AsyncMock(return_value=result)

    with patch("mistralai.client.Mistral", return_value=client):
        await run_agent("hi", timeout_ms=5000, durable=False, observability=False)

    # The per-turn request timeout is handed to the SDK runner (default None otherwise).
    _, kwargs = client.beta.conversations.run_async.call_args
    assert kwargs["timeout_ms"] == 5000


@pytest.mark.asyncio
async def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Mistral API key not found"):
        await run_agent("hi", durable=False, observability=False)


def test_final_text_concatenates_only_message_outputs():
    fcall = MagicMock()
    fcall.type = "function.call"
    entries = [_message("Hello "), fcall, _message("world")]
    assert _final_text(entries) == "Hello world"


@pytest.mark.asyncio
async def test_durable_turn_records_and_replays_a_conversation_response():
    from mistralai.client.models import ConversationResponse

    real = ConversationResponse.model_validate(
        {
            "conversation_id": "c1",
            "outputs": [{"type": "message.output", "content": "hi", "role": "assistant"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    conv = MagicMock()
    orig_start = AsyncMock(return_value=real)
    conv.start_async = orig_start
    conv.append_async = AsyncMock(return_value=real)

    _install_turn_hooks(conv, durable=True, usage=None)
    out = await conv.start_async(inputs="hi", model="m")  # now routed through durable_step

    orig_start.assert_awaited_once()
    # Round-tripped through pydantic JSON, polymorphic outputs intact.
    assert out.conversation_id == "c1"
    assert out.outputs[0].type == "message.output"


def _response_with_usage(prompt, completion, total):
    from mistralai.client.models import ConversationResponse

    return ConversationResponse.model_validate(
        {
            "conversation_id": "c1",
            "outputs": [{"type": "message.output", "content": "hi", "role": "assistant"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total},
        }
    )


def test_usage_sink_tallies_turns_and_tokens():
    sink = _UsageSink()
    sink.add(_response_with_usage(10, 5, 15))
    sink.add(_response_with_usage(20, 8, 28))

    assert sink.turns == 2
    assert (sink.prompt, sink.completion, sink.total) == (30, 13, 43)
    detail = sink.detail()
    assert "2 model turns" in detail and "43 total tokens" in detail
    assert "cached" not in detail  # a fresh run bills everything; nothing cached


def test_usage_sink_counts_cached_tokens_on_replay():
    # On a retry every turn is served from its durable record (no model call), so those
    # tokens are counted as cached — the row shows they weren't re-billed.
    sink = _UsageSink()
    sink.add(_response_with_usage(100, 20, 120), cached=True)
    sink.add(_response_with_usage(50, 10, 60), cached=True)

    assert sink.turns == 2 and sink.total == 180 and sink.cached == 180
    assert "180 cached" in sink.detail()


def test_usage_sink_partial_cache():
    # Crash mid-run: some turns come from cache, some are fresh — report the cached portion.
    sink = _UsageSink()
    sink.add(_response_with_usage(100, 20, 120), cached=True)
    sink.add(_response_with_usage(50, 10, 60), cached=False)

    assert sink.total == 180 and sink.cached == 120
    assert "120 cached" in sink.detail()


@pytest.mark.asyncio
async def test_turn_hooks_tally_usage_when_not_durable():
    # observability-only path: no durable_step, but every turn still flows through the
    # wrapper, so tokens are tallied from each ConversationResponse.usage.
    resp = _response_with_usage(7, 3, 10)
    conv = MagicMock()
    conv.start_async = AsyncMock(return_value=resp)
    conv.append_async = AsyncMock(return_value=resp)

    sink = _UsageSink()
    _install_turn_hooks(conv, durable=False, usage=sink)
    await conv.start_async(inputs="hi", model="m")
    await conv.append_async(inputs="more", model="m")

    assert sink.turns == 2
    assert sink.total == 20
