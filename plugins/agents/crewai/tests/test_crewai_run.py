"""Tests for CrewAI run_agent (mocked — no network)."""

import inspect

import flyte
import pytest
from crewai.tools import BaseTool

import flyteplugins.agents.crewai._run as run_mod
from flyteplugins.agents.crewai import tool


class _FakeOutput:
    """Stand-in for CrewAI's ``LiteAgentOutput``."""

    def __init__(self, raw):
        self.raw = raw


class _FakeAgent:
    """A minimal fake CrewAI agent."""

    def __init__(self, raw):
        self._raw = raw
        self.kicked_with = None

    async def kickoff_async(self, messages, **kwargs):
        self.kicked_with = messages
        return _FakeOutput(self._raw)


@pytest.mark.asyncio
async def test_run_agent_with_prebuilt_agent():
    """run_agent drives a pre-built agent via kickoff_async and extracts .raw."""
    agent = _FakeAgent("Hello!")
    result = await run_mod.run_agent("Hi", agent=agent, name="test-agent")
    assert result == "Hello!"
    assert agent.kicked_with == "Hi"


@pytest.mark.asyncio
async def test_run_agent_builds_agent_and_attaches_tools_natively(monkeypatch):
    """When no agent is passed, run_agent builds one with tools attached natively."""
    env = flyte.TaskEnvironment("crewai_run_a")

    @tool
    @env.task
    def get_weather(city: str) -> str:
        """Get weather."""
        return f"sunny in {city}"

    captured = {}

    class _BuiltAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.tools = kwargs.get("tools", [])

        async def kickoff_async(self, messages, **kwargs):
            return _FakeOutput("The weather is sunny.")

    # Patch the Agent class the builder imports.
    import crewai

    monkeypatch.setattr(crewai, "Agent", _BuiltAgent)

    result = await run_mod.run_agent(
        "What's the weather?",
        tools=[get_weather],
        model="gpt-4o",
        name="test-agent",
    )
    assert result == "The weather is sunny."
    # Tools were attached to the Agent constructor as native BaseTools.
    assert len(captured["tools"]) == 1
    assert all(isinstance(t, BaseTool) for t in captured["tools"])


@pytest.mark.asyncio
async def test_run_agent_raises_on_both_agent_and_tools():
    with pytest.raises(ValueError, match="Pass either"):
        await run_mod.run_agent("hi", agent=_FakeAgent("x"), tools=[lambda: None])


@pytest.mark.asyncio
async def test_run_agent_requires_model_on_builder_path():
    """No default model is assumed: the builder path demands an explicit `model=`."""
    with pytest.raises(ValueError, match="Provide `model=`"):
        await run_mod.run_agent("hi")


def test_run_agent_sync_variant():
    """run_agent is async; run_agent_sync runs it from synchronous code."""
    assert inspect.iscoroutinefunction(run_mod.run_agent)
    agent = _FakeAgent("sync!")
    assert run_mod.run_agent_sync("Hi", agent=agent) == "sync!"
