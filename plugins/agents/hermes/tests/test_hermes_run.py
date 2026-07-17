"""Tests for Hermes run_agent (mocked — no network)."""

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


@pytest.mark.asyncio
async def test_run_agent_durable_is_a_noop_for_hermes():
    """durable=True is a documented no-op: the agent is driven exactly once either way."""
    calls = {"n": 0}

    class _Once:
        def run_conversation(self, user_message, **kwargs):
            calls["n"] += 1
            return {"final_response": "answer"}

    assert await run_mod.run_agent("hi", agent=_Once(), durable=True) == "answer"
    assert calls["n"] == 1
    assert await run_mod.run_agent("hi", agent=_Once(), durable=False) == "answer"


def test_run_agent_sync_variant():
    """run_agent is async; run_agent_sync runs it from synchronous code."""
    import inspect

    assert inspect.iscoroutinefunction(run_mod.run_agent)
    # The sync variant actually drives the agent (no event loop in this test).
    assert run_mod.run_agent_sync("Hi", agent=_FakeAgent("sync!")) == "sync!"
