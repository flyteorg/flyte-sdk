"""Tests for Hermes run_agent (mocked)."""

import flyte
import pytest

import flyteplugins.agents.hermes._run as run_mod


class _FakeResult:
    """A minimal fake Hermes run result."""

    def __init__(self, data):
        self.data = data


class _FakeAgent:
    """A minimal fake Hermes agent."""

    def __init__(self, result):
        self._result = result

    async def run(self, message, **kwargs):
        return self._result


@pytest.mark.asyncio
async def test_run_agent_with_tools(monkeypatch):
    """run_agent builds an agent from tools and runs it."""
    env = flyte.TaskEnvironment("h_run_a")

    @env.task
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    fake_agent = _FakeAgent("The weather is sunny.")

    def fake_build(*args, **kwargs):
        return fake_agent

    monkeypatch.setattr(run_mod, "_Agent", fake_build)

    result = await run_mod.run_agent("What's the weather?", tools=[get_weather], name="test-agent")
    assert result is not None


@pytest.mark.asyncio
async def test_run_agent_with_prebuilt_agent(monkeypatch):
    """run_agent accepts a pre-built agent."""
    fake_agent = _FakeAgent("Hello!")

    result = await run_mod.run_agent("Hi", agent=fake_agent, name="test-agent")
    assert result is not None


@pytest.mark.asyncio
async def test_run_agent_raises_on_both_agent_and_tools():
    with pytest.raises(ValueError, match="Pass either"):
        await run_mod.run_agent("hi", agent=_FakeAgent("x"), tools=[lambda: None])


class _CapturingAgent:
    """Fake agent that records the kwargs it was driven with and counts calls."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def run(self, message, **kwargs):
        self.calls.append((message, kwargs))
        return self.reply


@pytest.mark.asyncio
async def test_run_agent_persists_and_resumes_memory(monkeypatch):
    """With a memory_key, the transcript is saved and replayed as message_history."""
    from tests.test_hermes_memory import _FakeStore

    store = _FakeStore()

    async def _resolve(key):
        return store if key else None

    monkeypatch.setattr(run_mod, "resolve_memory", _resolve)

    agent = _CapturingAgent("hello there")
    await run_mod.run_agent("first", agent=agent, memory_key="u1")
    await run_mod.run_agent("second", agent=agent, memory_key="u1")

    # The second run resumes: it receives the first turn's transcript as history.
    _, kwargs = agent.calls[1]
    assert kwargs["message_history"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "hello there"},
    ]


@pytest.mark.asyncio
async def test_run_agent_durable_records_once(monkeypatch):
    """durable=True drives the agent through the durable step (still runs once)."""
    calls = {"n": 0}

    class _Once:
        async def run(self, message, **kwargs):
            calls["n"] += 1
            return "answer"

    result = await run_mod.run_agent("hi", agent=_Once(), durable=True)
    assert result == "answer"
    assert calls["n"] == 1
