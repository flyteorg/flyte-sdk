"""Tests for the Mistral run_agent (SDK runner) + the durable-turn seam.

The agent loop itself is the SDK's (``conversations.run_async``), so these mock
it and verify our orchestration: tools are registered, the final text is
extracted, and the durable-turn wrapping records/replays a turn faithfully.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import flyte
import pytest

from flyteplugins.agents.mistral import function_tool, run_agent
from flyteplugins.agents.mistral._run import _final_text, _install_durable_turns


def _message(text):
    e = MagicMock()
    e.type = "message.output"
    e.content = text
    return e


@pytest.mark.asyncio
async def test_run_agent_registers_tools_and_returns_final_text(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    env = flyte.TaskEnvironment("mistral_run_a")

    @function_tool
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

    _install_durable_turns(conv)
    out = await conv.start_async(inputs="hi", model="m")  # now routed through durable_step

    orig_start.assert_awaited_once()
    # Round-tripped through pydantic JSON, polymorphic outputs intact.
    assert out.conversation_id == "c1"
    assert out.outputs[0].type == "message.output"
