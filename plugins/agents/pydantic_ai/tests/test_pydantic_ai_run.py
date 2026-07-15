"""Tests for Pydantic AI run_agent (mocked)."""

import flyte
import pytest

import flyteplugins.agents.pydantic_ai._run as run_mod


class _FakeResult:
    """A minimal fake Pydantic AI run result."""

    def __init__(self, data):
        self.data = data


class _FakeAgent:
    """A minimal fake Pydantic AI agent."""

    def __init__(self, result):
        self._result = result

    async def run(self, message, **kwargs):
        return self._result


@pytest.mark.asyncio
async def test_run_agent_with_tools(monkeypatch):
    """run_agent builds an agent from tools and runs it."""
    env = flyte.TaskEnvironment("pai_run_a")

    @env.task
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    fake_agent = _FakeAgent(_FakeResult("The weather is sunny."))

    def fake_build(*args, **kwargs):
        return fake_agent

    monkeypatch.setattr(run_mod, "_Agent", fake_build)

    result = await run_mod.run_agent("What's the weather?", tools=[get_weather], name="test-agent")
    assert result is not None


@pytest.mark.asyncio
async def test_run_agent_with_prebuilt_agent(monkeypatch):
    """run_agent accepts a pre-built agent."""
    fake_agent = _FakeAgent(_FakeResult("Hello!"))

    result = await run_mod.run_agent("Hi", agent=fake_agent, name="test-agent")
    assert result is not None


@pytest.mark.asyncio
async def test_run_agent_raises_on_both_agent_and_tools():
    with pytest.raises(ValueError, match="Pass either"):
        await run_mod.run_agent("hi", agent=_FakeAgent("x"), tools=[lambda: None])
