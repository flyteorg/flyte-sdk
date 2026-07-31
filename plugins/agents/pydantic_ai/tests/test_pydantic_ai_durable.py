"""Unit tests for the durable model-turn wrapper (no network)."""

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import Model, ModelRequestParameters

from flyteplugins.agents.pydantic_ai import FlyteModel
from flyteplugins.agents.pydantic_ai._durable import _request_fingerprint


class _FakeTool:
    def __init__(self, name: str):
        self.name = name


class _InnerModel(Model):
    """A canned inner model returning a fixed ModelResponse (no network)."""

    def __init__(self, response: ModelResponse):
        self._response = response
        self.calls = 0

    async def request(self, messages, model_settings, model_request_parameters):
        self.calls += 1
        return self._response

    def request_stream(self, *args, **kwargs):  # pragma: no cover - not exercised
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def system(self) -> str:
        return "fake-system"


@pytest.mark.asyncio
async def test_request_roundtrips_through_json_outside_task_context():
    """Outside a task context flyte.trace is transparent; the turn is serialized to
    JSON and rebuilt, so the inner model is called once and the rebuilt
    ModelResponse matches the original."""
    resp = ModelResponse(parts=[TextPart(content="Hello")])
    inner = _InnerModel(resp)
    model = FlyteModel(inner)

    messages = [ModelRequest(parts=[UserPromptPart(content="hi")])]
    out = await model.request(messages, None, ModelRequestParameters())

    assert inner.calls == 1
    assert isinstance(out, ModelResponse)
    assert out.parts[0].content == "Hello"


def test_flyte_model_forwards_name_and_system():
    inner = _InnerModel(ModelResponse(parts=[]))
    model = FlyteModel(inner)
    assert model.model_name == "fake-model"
    assert model.system == "fake-system"


def test_fingerprint_is_deterministic():
    messages = [ModelRequest(parts=[UserPromptPart(content="hello")])]
    params = ModelRequestParameters()
    assert _request_fingerprint(messages, None, params) == _request_fingerprint(messages, None, params)


def test_fingerprint_changes_with_input():
    params = ModelRequestParameters()
    a = [ModelRequest(parts=[UserPromptPart(content="hello")])]
    b = [ModelRequest(parts=[UserPromptPart(content="different")])]
    assert _request_fingerprint(a, None, params) != _request_fingerprint(b, None, params)


def test_fingerprint_is_tool_order_insensitive():
    messages = [ModelRequest(parts=[UserPromptPart(content="hello")])]
    p1 = ModelRequestParameters(function_tools=[_FakeTool("a"), _FakeTool("b")])
    p2 = ModelRequestParameters(function_tools=[_FakeTool("b"), _FakeTool("a")])
    assert _request_fingerprint(messages, None, p1) == _request_fingerprint(messages, None, p2)
