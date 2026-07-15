"""Tests for LangChain run_agent (mocked)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import flyte
import pytest

import flyteplugins.agents.langchain._run as run_mod


class _FakeAgent:
    """A minimal fake compiled ``create_agent`` graph."""

    def __init__(self, result):
        self._result = result

    async def ainvoke(self, input_dict, **kwargs):
        return self._result


@pytest.mark.asyncio
async def test_run_agent_with_tools(monkeypatch):
    """run_agent builds an agent from tools (via create_agent) and runs it."""
    env = flyte.TaskEnvironment("lc_run_a")

    @env.task
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    # In langchain 1.x the agent is a compiled graph returning a messages state.
    fake_agent = _FakeAgent({"messages": [SimpleNamespace(content="The weather is sunny.")]})

    def fake_create_agent(model, tools, *, system_prompt=None, **kwargs):
        return fake_agent

    mock_chat_openai = MagicMock(spec=[])
    # Patch _ChatOpenAI so the real ChatOpenAI import is skipped entirely, and
    # _create_agent so no real graph (or model) is constructed.
    monkeypatch.setattr(run_mod, "_ChatOpenAI", mock_chat_openai)
    monkeypatch.setattr(run_mod, "_create_agent", fake_create_agent)

    result = await run_mod.run_agent("What's the weather?", tools=[get_weather], name="test-agent")
    assert result == "The weather is sunny."


@pytest.mark.asyncio
async def test_run_agent_with_prebuilt_agent(monkeypatch):
    """run_agent accepts a pre-built agent."""
    fake_agent = _FakeAgent({"output": "Hello!"})

    result = await run_mod.run_agent("Hi", agent=fake_agent, name="test-agent")
    assert result is not None


@pytest.mark.asyncio
async def test_run_agent_raises_on_both_agent_and_tools():
    with pytest.raises(ValueError, match="Pass either"):
        await run_mod.run_agent("hi", agent=_FakeAgent({}), tools=[lambda: None])


class _Aio:
    def __init__(self, fn):
        self._fn = fn

    async def aio(self, *a, **k):
        return self._fn(*a, **k)


class _FakeStore:
    def __init__(self):
        self.data: dict = {}
        self.messages: list = []
        self.read_json = _Aio(lambda path, default=None: self.data.get(path, default))
        self.write_json = _Aio(lambda path, obj, **kw: self.data.__setitem__(path, obj))
        self.save = _Aio(lambda: None)
        self.extend = self.messages.extend


@pytest.mark.asyncio
async def test_run_agent_memory_loads_prepends_and_saves(monkeypatch):
    """With memory_key, prior history is prepended and the full transcript is saved."""
    from langchain_core.messages import AIMessage, HumanMessage

    store = _FakeStore()
    # Seed prior history in the store.
    from flyteplugins.agents.langchain import _memory as memory_mod

    await memory_mod.save_history(store, [HumanMessage(content="earlier"), AIMessage(content="reply")])

    monkeypatch.setattr(run_mod, "resolve_memory", lambda key: _await_value(store))

    seen = {}

    class _RecordingAgent:
        async def ainvoke(self, input_dict, **kwargs):
            seen["messages"] = input_dict["messages"]
            # Mirror create_agent: normalize dict inputs into message objects.
            normalized = [
                HumanMessage(content=m["content"]) if isinstance(m, dict) else m for m in input_dict["messages"]
            ]
            return {"messages": [*normalized, AIMessage(content="final answer")]}

    def fake_create_agent(model, tools, *, system_prompt=None, **kwargs):
        return _RecordingAgent()

    mock_chat_openai = MagicMock(spec=[])
    monkeypatch.setattr(run_mod, "_ChatOpenAI", mock_chat_openai)
    monkeypatch.setattr(run_mod, "_create_agent", fake_create_agent)

    result = await run_mod.run_agent("new question", tools=[], memory_key="user-1", durable=False)
    assert result == "final answer"

    # Prior messages were prepended, followed by the new user turn.
    passed = seen["messages"]
    assert getattr(passed[0], "content", passed[0]) == "earlier"
    assert passed[-1] == {"role": "user", "content": "new question"}

    # The full transcript (prior + new + final) was written back.
    saved = await memory_mod.load_history(store)
    assert [m.content for m in saved] == ["earlier", "reply", "new question", "final answer"]


@pytest.mark.asyncio
async def test_run_agent_wraps_model_when_durable(monkeypatch):
    """On the builder path with durable=True, the inner model is wrapped in DurableChatModel."""
    from langchain_core.messages import AIMessage

    captured = {}

    def fake_create_agent(model, tools, *, system_prompt=None, **kwargs):
        captured["model"] = model
        return _FakeAgent({"messages": [AIMessage(content="ok")]})

    # A real BaseChatModel so the isinstance check in _wrap_durable passes.
    from tests.test_langchain_durable import _FakeInner

    monkeypatch.setattr(run_mod, "_create_agent", fake_create_agent)

    result = await run_mod.run_agent("hi", tools=[], model=_FakeInner(), durable=True)
    assert result == "ok"

    from flyteplugins.agents.langchain._durable import DurableChatModel

    assert isinstance(captured["model"], DurableChatModel)


@pytest.mark.asyncio
async def test_run_agent_does_not_wrap_when_durable_false(monkeypatch):
    from langchain_core.messages import AIMessage

    from tests.test_langchain_durable import _FakeInner

    captured = {}

    def fake_create_agent(model, tools, *, system_prompt=None, **kwargs):
        captured["model"] = model
        return _FakeAgent({"messages": [AIMessage(content="ok")]})

    monkeypatch.setattr(run_mod, "_create_agent", fake_create_agent)

    inner = _FakeInner()
    await run_mod.run_agent("hi", tools=[], model=inner, durable=False)
    assert captured["model"] is inner


def _await_value(value):
    async def _coro():
        return value

    return _coro()
