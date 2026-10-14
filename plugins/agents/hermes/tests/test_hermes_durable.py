"""Tests for durable (record/replay) Hermes model turns (mocked — no network)."""

import types

import pytest

from flyteplugins.agents.hermes import _durable


def _chat_completion_namespace():
    """A ChatCompletion-shaped response carrying a tool call, as SimpleNamespace."""
    return types.SimpleNamespace(
        id="chatcmpl-1",
        model="test-model",
        choices=[
            types.SimpleNamespace(
                index=0,
                finish_reason="tool_calls",
                message=types.SimpleNamespace(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        types.SimpleNamespace(
                            id="call_1",
                            type="function",
                            function=types.SimpleNamespace(name="get_weather", arguments='{"city": "Paris"}'),
                        )
                    ],
                ),
            )
        ],
        usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=3, total_tokens=13),
    )


@pytest.fixture
def durable_gate():
    """Open the durable gate for the test and reset it afterwards."""
    token = _durable.enable_durable()
    try:
        yield
    finally:
        _durable.disable_durable(token)


@pytest.fixture
def clean_middleware():
    """Restore the Hermes middleware list after a test registers into it."""
    from hermes_cli.plugins import get_plugin_manager

    callbacks = get_plugin_manager()._middleware.setdefault("llm_execution", [])
    before = list(callbacks)
    try:
        yield callbacks
    finally:
        callbacks[:] = before


def test_durable_callback_calls_next_call_once_and_round_trips(durable_gate):
    """The real provider call happens once; the result survives dumps/loads."""
    calls = {"n": 0}
    response = _chat_completion_namespace()

    def next_call(request):
        calls["n"] += 1
        return response

    request = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "weather in Paris?"}],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
    }
    out = _durable.durable_llm_execution(request, next_call, api_mode="chat_completions", api_call_count=1)

    assert calls["n"] == 1
    assert out.model == "test-model"
    assert out.choices[0].finish_reason == "tool_calls"
    assert out.choices[0].message.content is None
    tool_call = out.choices[0].message.tool_calls[0]
    assert tool_call.id == "call_1"
    assert tool_call.function.name == "get_weather"
    assert tool_call.function.arguments == '{"city": "Paris"}'
    assert out.usage.total_tokens == 13


def test_callback_is_passthrough_without_the_gate():
    """With the contextvar unset the middleware returns the provider object untouched."""
    sentinel = object()
    seen = {}

    def next_call(request):
        seen["request"] = request
        return sentinel

    request = {"model": "m", "messages": []}
    assert _durable.durable_llm_execution(request, next_call) is sentinel
    assert seen["request"] is request


def test_ensure_registered_is_idempotent(clean_middleware):
    """Registering twice leaves exactly one durable callback in the chain."""
    _durable.ensure_registered()
    _durable.ensure_registered()

    registered = [cb for cb in clean_middleware if cb is _durable.durable_llm_execution]
    assert registered == [_durable.durable_llm_execution]


def test_request_key_varies_with_api_call_count():
    """Two textually identical requests in one run stay distinct steps."""
    request = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    first = _durable._request_key(request, {"api_mode": "chat_completions", "api_call_count": 1})
    second = _durable._request_key(request, {"api_mode": "chat_completions", "api_call_count": 2})

    assert first != second
    assert first == _durable._request_key(request, {"api_mode": "chat_completions", "api_call_count": 1})


def test_request_key_tolerates_live_objects():
    """Non-serializable members in the request never break fingerprinting."""
    request = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "t"}}],
        "client": object(),
    }
    assert isinstance(_durable._request_key(request, {"api_mode": "chat_completions"}), str)


def test_anthropic_shape_round_trips(durable_gate):
    """An anthropic_messages response keeps its content blocks through replay."""
    response = types.SimpleNamespace(
        id="msg_1",
        model="claude-test",
        stop_reason="tool_use",
        content=[
            types.SimpleNamespace(type="text", text="Let me check."),
            types.SimpleNamespace(type="tool_use", id="tu_1", name="get_weather", input={"city": "Paris"}),
        ],
    )
    request = {
        "model": "claude-test",
        "system": "Be terse.",
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"name": "get_weather"}],
    }
    out = _durable.durable_llm_execution(request, lambda r: response, api_mode="anthropic_messages")

    assert out.stop_reason == "tool_use"
    assert out.content[0].type == "text"
    assert out.content[1].input.city == "Paris"


def test_pydantic_chat_completion_round_trips():
    """A real openai ChatCompletion serializes and rebuilds with its tool calls intact."""
    openai_types = pytest.importorskip("openai.types.chat")

    completion = openai_types.ChatCompletion.model_validate(
        {
            "id": "chatcmpl-2",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4.1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {"name": "get_population", "arguments": '{"city": "Tokyo"}'},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
    )

    rebuilt = _durable._loads(_durable._dumps(completion))

    assert rebuilt.model == "gpt-4.1"
    assert rebuilt.choices[0].finish_reason == "tool_calls"
    assert rebuilt.choices[0].message.tool_calls[0].function.arguments == '{"city": "Tokyo"}'
    assert rebuilt.usage.total_tokens == 7


def test_tool_names_are_sorted_and_shape_agnostic():
    """Tool names are extracted from both the OpenAI and Anthropic request shapes."""
    openai_shape = {"tools": [{"function": {"name": "b"}}, {"function": {"name": "a"}}]}
    anthropic_shape = {"tools": [{"name": "b"}, {"name": "a"}]}

    assert _durable._tool_names(openai_shape) == ["a", "b"]
    assert _durable._tool_names(anthropic_shape) == ["a", "b"]
    assert _durable._tool_names({}) == []
