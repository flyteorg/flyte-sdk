"""Tests for LangGraph run_agent (mocked)."""

import flyte
import pytest

import flyteplugins.agents.langgraph._run as run_mod


class _FakeAgent:
    """A minimal fake compiled LangGraph agent."""

    def __init__(self, result):
        self._result = result

    async def ainvoke(self, state, **kwargs):
        return self._result


@pytest.mark.asyncio
async def test_run_agent_with_tools(monkeypatch):
    """run_agent builds a graph from tools and runs it."""
    env = flyte.TaskEnvironment("lg_run_a")

    @env.task
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    fake_agent = _FakeAgent({"messages": [{"content": "The weather is sunny."}]})

    class FakeStateGraph:
        def __init__(self, *args, **kwargs):
            pass

        def add_node(self, *a, **kw):
            pass

        def add_edge(self, *a, **kw):
            pass

        def set_entry_point(self, *a, **kw):
            pass

        def add_conditional_edges(self, *a, **kw):
            pass

        def compile(self, *a, **kw):
            return fake_agent

    monkeypatch.setattr(run_mod, "_StateGraph", FakeStateGraph)

    result = await run_mod.run_agent("What's the weather?", tools=[get_weather], name="test-agent")
    assert result is not None


@pytest.mark.asyncio
async def test_run_agent_with_prebuilt_agent(monkeypatch):
    """run_agent accepts a pre-built agent."""
    fake_agent = _FakeAgent({"messages": [{"content": "Hello!"}]})

    result = await run_mod.run_agent("Hi", agent=fake_agent, name="test-agent")
    assert result is not None


@pytest.mark.asyncio
async def test_run_agent_raises_on_both_agent_and_tools():
    with pytest.raises(ValueError, match="Pass either"):
        await run_mod.run_agent("hi", agent=_FakeAgent({}), tools=[lambda: None])
