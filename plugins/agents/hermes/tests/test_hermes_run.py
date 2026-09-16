"""Tests for Hermes run_agent (mocked — no network)."""

import types

import flyte
import pytest

import flyteplugins.agents.hermes._run as run_mod


class _FakeAgent:
    """A minimal fake Hermes ``AIAgent``: sync ``run_conversation`` returning a dict."""

    def __init__(self, reply):
        self._reply = reply
        self.calls = []

    def run_conversation(self, user_message, **kwargs):
        self.calls.append((user_message, kwargs))
        return {"final_response": self._reply}


@pytest.mark.asyncio
async def test_run_agent_with_tools_builds_agent(monkeypatch):
    """run_agent builds an AIAgent from tools + model and drives it."""
    env = flyte.TaskEnvironment("h_run_a")

    @env.task
    def get_weather(city: str) -> str:
        """Get weather."""
        return f"sunny in {city}"

    captured = {}

    class _FakeAIAgent(_FakeAgent):
        def __init__(self, **kwargs):
            super().__init__("The weather is sunny.")
            captured.update(kwargs)

    monkeypatch.setattr(run_mod, "_AIAgent", _FakeAIAgent)

    result = await run_mod.run_agent("What's the weather?", tools=[get_weather], model="test-model", name="test-agent")
    assert result == "The weather is sunny."
    assert captured["model"] == "test-model"
    assert captured["quiet_mode"] is True
    assert captured["enabled_toolsets"] == ["flyte-test-agent"]
    assert "helpful assistant" in captured["ephemeral_system_prompt"]


@pytest.mark.asyncio
async def test_run_agent_requires_model_on_builder_path():
    """No default model: the builder path without `model=` is an error."""
    with pytest.raises(ValueError, match="Provide `model=`"):
        await run_mod.run_agent("hi", tools=[])


@pytest.mark.asyncio
async def test_run_agent_with_prebuilt_agent():
    """run_agent accepts a pre-built agent; instructions become the system message."""
    agent = _FakeAgent("Hello!")

    result = await run_mod.run_agent("Hi", agent=agent, instructions="Be terse.")
    assert result == "Hello!"
    _, kwargs = agent.calls[0]
    assert kwargs["system_message"] == "Be terse."


@pytest.mark.asyncio
async def test_run_agent_tolerates_plain_string_result():
    """A fake/wrapper returning a bare string (not a dict) still works."""

    class _Bare:
        def run_conversation(self, user_message, **kwargs):
            return "plain answer"

    assert await run_mod.run_agent("hi", agent=_Bare()) == "plain answer"


@pytest.mark.asyncio
async def test_run_agent_raises_on_both_agent_and_tools():
    with pytest.raises(ValueError, match="Pass either"):
        await run_mod.run_agent("hi", agent=_FakeAgent("x"), tools=[lambda: None])


@pytest.mark.asyncio
async def test_run_agent_raises_on_agent_kwargs_with_prebuilt_agent():
    with pytest.raises(ValueError, match="agent_kwargs"):
        await run_mod.run_agent("hi", agent=_FakeAgent("x"), api_key="sk-test")


@pytest.mark.asyncio
async def test_run_agent_persists_and_resumes_memory(monkeypatch):
    """With a memory_key, the transcript is saved and replayed as conversation_history."""
    from tests.test_hermes_memory import _FakeStore

    store = _FakeStore()

    async def _resolve(key):
        return store if key else None

    monkeypatch.setattr(run_mod, "resolve_memory", _resolve)

    agent = _FakeAgent("hello there")
    await run_mod.run_agent("first", agent=agent, memory_key="u1")
    await run_mod.run_agent("second", agent=agent, memory_key="u1")

    # The second run resumes: it receives the first turn's transcript as history.
    _, kwargs = agent.calls[1]
    assert kwargs["conversation_history"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "hello there"},
    ]


class _MiddlewareAgent:
    """A fake Hermes agent whose turn goes through the real `llm_execution` chain.

    Mirrors what `agent.conversation_loop` does: hand the provider call to
    `run_llm_execution_middleware` and consume the response by attribute.
    """

    def __init__(self):
        self.provider_calls = 0
        self.raw_responses = []
        self.responses = []

    def run_conversation(self, user_message, **kwargs):
        from hermes_cli.middleware import run_llm_execution_middleware

        def _perform_api_call(request):
            self.provider_calls += 1
            raw = types.SimpleNamespace(
                model="test-model",
                choices=[
                    types.SimpleNamespace(
                        finish_reason="stop",
                        message=types.SimpleNamespace(role="assistant", content="answer", tool_calls=None),
                    )
                ],
            )
            self.raw_responses.append(raw)
            return raw

        response = run_llm_execution_middleware(
            {"model": "test-model", "messages": [{"role": "user", "content": user_message}]},
            _perform_api_call,
            api_mode="chat_completions",
            api_call_count=1,
        )
        self.responses.append(response)
        return {"final_response": response.choices[0].message.content}


@pytest.fixture
def clean_middleware():
    """Restore the process-global Hermes middleware list after the test."""
    from hermes_cli.plugins import get_plugin_manager

    callbacks = get_plugin_manager()._middleware.setdefault("llm_execution", [])
    before = list(callbacks)
    try:
        yield callbacks
    finally:
        callbacks[:] = before


@pytest.mark.asyncio
async def test_run_agent_durable_records_model_turns(clean_middleware):
    """durable=True registers the middleware and routes the turn through it."""
    from flyteplugins.agents.hermes import _durable

    agent = _MiddlewareAgent()
    assert await run_mod.run_agent("hi", agent=agent, durable=True) == "answer"

    # Registered exactly once, and the provider was called exactly once.
    registered = [cb for cb in clean_middleware if cb is _durable.durable_llm_execution]
    assert registered == [_durable.durable_llm_execution]
    assert agent.provider_calls == 1
    # Engaged: the response the agent saw is the rebuilt, round-tripped shape,
    # not the SimpleNamespace the fake provider returned.
    assert agent.responses[0].choices[0].message.content == "answer"
    assert agent.responses[0] is not agent.raw_responses[0]
    assert agent.responses[0].choices[0].message.tool_calls is None

    # The gate is closed again once the run finishes.
    assert _durable._DURABLE.get() is False


@pytest.mark.asyncio
async def test_run_agent_not_durable_leaves_middleware_disengaged(clean_middleware):
    """durable=False leaves the gate closed, so the turn is not recorded.

    Registration itself is process-global and idempotent, so the callback may
    already sit in the chain from another run; what makes this run non-durable
    is the closed gate, which turns the callback into a passthrough.
    """
    from flyteplugins.agents.hermes import _durable

    _durable.ensure_registered()
    agent = _MiddlewareAgent()
    assert await run_mod.run_agent("hi", agent=agent, durable=False) == "answer"

    assert agent.provider_calls == 1
    # Untouched: the agent saw the provider object itself, not a rebuilt one.
    assert agent.responses[0] is agent.raw_responses[0]
    assert _durable._DURABLE.get() is False


def test_run_agent_sync_variant():
    """run_agent is async; run_agent_sync runs it from synchronous code."""
    import inspect

    assert inspect.iscoroutinefunction(run_mod.run_agent)
    # The sync variant actually drives the agent (no event loop in this test).
    assert run_mod.run_agent_sync("Hi", agent=_FakeAgent("sync!")) == "sync!"
