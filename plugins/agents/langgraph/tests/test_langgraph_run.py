"""Tests for LangGraph run_agent (mocked — no network / no controller)."""

import flyte
import pytest

import flyteplugins.agents.langgraph._run as run_mod


class _FakeAgent:
    """A minimal fake compiled LangGraph graph."""

    def __init__(self, result):
        self._result = result
        self.calls = []

    async def ainvoke(self, state, **kwargs):
        self.calls.append(state)
        return self._result


class _FakeModel:
    """A chat model stub whose ``bind_tools`` is a no-op."""

    def bind_tools(self, tools):
        return self


@pytest.mark.asyncio
async def test_run_agent_with_tools_builds_default_graph(monkeypatch):
    """run_agent builds the default graph from tools and runs it."""
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

        def add_conditional_edges(self, *a, **kw):
            pass

        def compile(self, *a, **kw):
            return fake_agent

    # Redirect the builder to a fake graph + fake model so nothing touches OpenAI.
    monkeypatch.setattr(run_mod, "_StateGraph", FakeStateGraph)
    monkeypatch.setattr(run_mod, "_resolve_chat_model", lambda model: _FakeModel())

    result = await run_mod.run_agent("What's the weather?", tools=[get_weather], name="test-agent")
    assert result == "The weather is sunny."
    # The default builder seeds a messages state.
    assert "messages" in fake_agent.calls[0]


@pytest.mark.asyncio
async def test_run_agent_with_prebuilt_agent():
    """run_agent drives a pre-built compiled graph and extracts the final text."""
    fake_agent = _FakeAgent({"messages": [{"content": "Hello!"}]})

    result = await run_mod.run_agent("Hi", agent=fake_agent, name="test-agent")
    assert result == "Hello!"


@pytest.mark.asyncio
async def test_run_agent_raises_on_both_agent_and_tools():
    with pytest.raises(ValueError, match="Pass either"):
        await run_mod.run_agent("hi", agent=_FakeAgent({}), tools=[lambda: None])


@pytest.mark.asyncio
async def test_run_agent_persists_and_resumes_memory(monkeypatch):
    """With a memory_key, the transcript is saved and prepended on the next run."""
    from langchain_core.messages import AIMessage

    import flyteplugins.agents.langgraph._memory as memory_mod
    from tests.test_langgraph_memory import _FakeStore

    store = _FakeStore()
    monkeypatch.setattr(memory_mod, "resolve_memory", lambda key: _returns(store if key else None))

    class _EchoAgent:
        """Appends an assistant reply to whatever messages it receives."""

        def __init__(self):
            self.seen = []

        async def ainvoke(self, state, **kwargs):
            msgs = list(state["messages"])
            self.seen.append(msgs)
            return {"messages": [*msgs, AIMessage(content="reply")]}

    agent = _EchoAgent()
    await run_mod.run_agent("first", agent=agent, memory_key="u1")
    # Second run resumes: the prior transcript is prepended before the new input.
    await run_mod.run_agent("second", agent=agent, memory_key="u1")

    contents = [m.content for m in agent.seen[1]]
    assert contents == ["first", "reply", "second"]


async def _returns(value):
    return value
