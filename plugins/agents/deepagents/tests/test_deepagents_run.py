"""Tests for deepagents run_agent (mocked — no network / no controller)."""

from types import SimpleNamespace

import flyte
import pytest

import flyteplugins.agents.deepagents._run as run_mod


class _FakeAgent:
    """A minimal fake compiled ``create_deep_agent`` graph."""

    def __init__(self, result):
        self._result = result
        self.calls = []

    async def ainvoke(self, state, **kwargs):
        self.calls.append(state)
        return self._result


@pytest.mark.asyncio
async def test_run_agent_with_tools(monkeypatch):
    """run_agent builds a deep agent from tools and runs it."""
    env = flyte.TaskEnvironment("da_run_a")

    @env.task
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    fake_agent = _FakeAgent({"messages": [SimpleNamespace(content="The weather is sunny.")], "files": {}})
    captured = {}

    def fake_create_deep_agent(*, model=None, tools=None, system_prompt=None, **kwargs):
        captured["model"] = model
        captured["tools"] = tools
        return fake_agent

    # Substitute the graph builder and the model resolver so no provider import
    # (or API key) is needed.
    monkeypatch.setattr(run_mod, "_create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(run_mod, "_resolve_chat_model", lambda model: object())

    result = await run_mod.run_agent.aio("What's the weather?", tools=[get_weather], name="test-agent")
    assert result == "The weather is sunny."
    # The bare task was coerced into a LangChain tool.
    assert captured["tools"][0].name == "get_weather"


@pytest.mark.asyncio
async def test_run_agent_with_prebuilt_agent():
    """run_agent drives a pre-built compiled graph and extracts the final text."""
    fake_agent = _FakeAgent({"messages": [SimpleNamespace(content="Hello!")], "files": {}})

    result = await run_mod.run_agent.aio("Hi", agent=fake_agent, name="test-agent")
    assert result == "Hello!"


def test_run_agent_is_syncified():
    """run_agent is callable synchronously, with an `.aio` async variant."""
    fake_agent = _FakeAgent({"messages": [SimpleNamespace(content="Hello!")], "files": {}})
    assert run_mod.run_agent("Hi", agent=fake_agent, name="test-agent") == "Hello!"


@pytest.mark.asyncio
async def test_run_agent_raises_on_both_agent_and_tools():
    with pytest.raises(ValueError, match="Pass either"):
        await run_mod.run_agent.aio("hi", agent=_FakeAgent({}), tools=[lambda: None])


@pytest.mark.asyncio
async def test_run_agent_requires_model_on_builder_path(monkeypatch):
    """Building an agent (no `agent=`) without a model is an explicit error."""
    monkeypatch.setattr(run_mod, "_create_deep_agent", lambda **kwargs: _FakeAgent({}))
    with pytest.raises(ValueError, match="Provide `model=`"):
        await run_mod.run_agent.aio("hi", tools=[])


@pytest.mark.asyncio
async def test_run_agent_memory_resumes_conversation_and_files(monkeypatch):
    """With memory_key, prior history + files are seeded and the updated state is saved."""
    from langchain_core.messages import AIMessage, HumanMessage

    import flyteplugins.agents.deepagents._memory as memory_mod
    from tests.test_deepagents_memory import _FakeStore

    store = _FakeStore()
    await memory_mod.save_state(
        store,
        [HumanMessage(content="earlier"), AIMessage(content="reply")],
        files={"notes.txt": "v1"},
    )

    monkeypatch.setattr(run_mod, "resolve_memory", lambda key: _await_value(store if key else None))

    class _RecordingAgent:
        def __init__(self):
            self.seen = None

        async def ainvoke(self, state, **kwargs):
            self.seen = state
            normalized = [HumanMessage(content=m["content"]) if isinstance(m, dict) else m for m in state["messages"]]
            return {
                "messages": [*normalized, AIMessage(content="final answer")],
                "files": {**state.get("files", {}), "notes.txt": "v2"},
            }

    agent = _RecordingAgent()
    result = await run_mod.run_agent.aio("new question", agent=agent, memory_key="user-1")
    assert result == "final answer"

    # Prior conversation and files were seeded into the run.
    assert getattr(agent.seen["messages"][0], "content", None) == "earlier"
    assert agent.seen["messages"][-1] == {"role": "user", "content": "new question"}
    assert agent.seen["files"] == {"notes.txt": "v1"}

    # The full transcript and the updated files were written back.
    saved = await memory_mod.load_history(store)
    assert [m.content for m in saved] == ["earlier", "reply", "new question", "final answer"]
    assert await memory_mod.load_files(store) == {"notes.txt": "v2"}


@pytest.mark.asyncio
async def test_run_agent_wraps_model_when_durable(monkeypatch):
    """On the builder path with durable=True, the model is wrapped in DurableChatModel."""
    from langchain_core.messages import AIMessage

    from flyteplugins.agents.deepagents._durable import DurableChatModel
    from tests.test_deepagents_durable import _FakeInner

    captured = {}

    def fake_create_deep_agent(*, model=None, tools=None, system_prompt=None, **kwargs):
        captured["model"] = model
        return _FakeAgent({"messages": [AIMessage(content="ok")], "files": {}})

    monkeypatch.setattr(run_mod, "_create_deep_agent", fake_create_deep_agent)

    inner = _FakeInner()
    result = await run_mod.run_agent.aio("hi", tools=[], model=inner, durable=True)
    assert result == "ok"
    assert isinstance(captured["model"], DurableChatModel)

    # durable=False passes the model through unwrapped.
    await run_mod.run_agent.aio("hi", tools=[], model=inner, durable=False)
    assert captured["model"] is inner


@pytest.mark.asyncio
async def test_run_agent_forwards_deepagents_kwargs(monkeypatch):
    """Deep-Agents-specific options (subagents=...) pass through to create_deep_agent."""
    from langchain_core.messages import AIMessage

    from tests.test_deepagents_durable import _FakeInner

    captured = {}

    def fake_create_deep_agent(*, model=None, tools=None, system_prompt=None, **kwargs):
        captured.update(kwargs)
        return _FakeAgent({"messages": [AIMessage(content="ok")], "files": {}})

    monkeypatch.setattr(run_mod, "_create_deep_agent", fake_create_deep_agent)

    subagents = [{"name": "critic", "description": "d", "system_prompt": "p"}]
    await run_mod.run_agent.aio("hi", tools=[], model=_FakeInner(), subagents=subagents)
    assert captured["subagents"] == subagents


def _await_value(value):
    async def _coro():
        return value

    return _coro()
