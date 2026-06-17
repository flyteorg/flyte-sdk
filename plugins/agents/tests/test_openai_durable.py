"""Unit tests for the durable model provider/model wrapper."""

import pytest
from agents.models.interface import Model

from flyteplugins.agents.openai import FlyteModel, FlyteModelProvider
from flyteplugins.agents.openai._durable import _bind, _fingerprint


class _FakeTool:
    def __init__(self, name: str):
        self.name = name


class _InnerModel(Model):
    def __init__(self, response):
        self._response = response
        self.calls = 0

    async def get_response(self, *args, **kwargs):
        self.calls += 1
        return self._response

    def stream_response(self, *args, **kwargs):  # pragma: no cover - not exercised
        raise NotImplementedError


@pytest.mark.asyncio
async def test_get_response_roundtrips_through_json_outside_task_context():
    """Outside a task context flyte.trace is transparent; the turn is serialized
    to JSON and rebuilt, so the inner model is called once and the rebuilt
    ModelResponse matches the original."""
    from agents.items import ModelResponse
    from agents.usage import Usage
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    resp = ModelResponse(
        output=[
            ResponseOutputMessage(
                id="m1",
                type="message",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(type="output_text", text="Hello", annotations=[])],
            )
        ],
        usage=Usage(),
        response_id="r1",
    )
    inner = _InnerModel(resp)
    model = FlyteModel(inner)
    out = await model.get_response(
        "system",
        "hi",
        None,
        [],
        None,
        [],
        None,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )
    assert inner.calls == 1
    assert isinstance(out, ModelResponse)
    assert out.response_id == "r1"
    assert out.output[0].content[0].text == "Hello"


def test_fingerprint_is_deterministic_and_tool_order_insensitive():
    b1 = _bind(("system", "hello", None, [_FakeTool("a"), _FakeTool("b")], None, []), {})
    b2 = _bind(("system", "hello", None, [_FakeTool("b"), _FakeTool("a")], None, []), {})
    assert _fingerprint(b1) == _fingerprint(b2)


def test_fingerprint_changes_with_input():
    base = _bind(("system", "hello", None, [], None, []), {})
    other = _bind(("system", "different", None, [], None, []), {})
    assert _fingerprint(base) != _fingerprint(other)


def test_provider_wraps_returned_models():
    class _Provider:
        def get_model(self, name):
            return _InnerModel("X")

    provider = FlyteModelProvider(_Provider())
    model = provider.get_model("gpt-4.1")
    assert isinstance(model, FlyteModel)
