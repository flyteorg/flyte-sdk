"""Tests for Google ADK durable model turns (``FlyteLlm`` over the model seam)."""

import pytest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as gt

from flyteplugins.agents.google._durable import FlyteLlm, durable_model


class _MockInner:
    """A stand-in ``BaseLlm`` that records how many times it was actually called."""

    model = "gemini-2.0-flash"

    def __init__(self):
        self.calls = 0

    async def generate_content_async(self, llm_request, stream=False):
        self.calls += 1
        yield LlmResponse(content=gt.Content(role="model", parts=[gt.Part.from_text(text="42")]))


class _Req:
    def model_dump(self, **kwargs):
        return {"contents": [{"role": "user", "parts": [{"text": "q"}]}]}


@pytest.mark.asyncio
async def test_flyte_llm_records_and_replays_a_turn():
    inner = _MockInner()
    llm = FlyteLlm(model="gemini-2.0-flash", inner=inner)

    out = [r async for r in llm.generate_content_async(_Req(), stream=False)]

    assert len(out) == 1
    # The response round-trips through the durable_step JSON record.
    assert out[0].content.parts[0].text == "42"
    assert inner.calls == 1


@pytest.mark.asyncio
async def test_flyte_llm_streaming_passes_through():
    inner = _MockInner()
    llm = FlyteLlm(model="gemini-2.0-flash", inner=inner)

    out = [r async for r in llm.generate_content_async(_Req(), stream=True)]

    assert out[0].content.parts[0].text == "42"


def test_durable_model_wraps_a_string_model():
    wrapped = durable_model("gemini-2.0-flash")
    assert type(wrapped).__name__ == "FlyteLlm"
    assert wrapped.model == "gemini-2.0-flash"
    assert wrapped.inner is not None
