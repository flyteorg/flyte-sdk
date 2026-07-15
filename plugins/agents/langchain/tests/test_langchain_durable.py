"""Tests for LangChain model-turn durability (no network).

Uses a fake inner ``BaseChatModel`` that returns canned ``AIMessage``s (including
one with tool calls) from ``_agenerate``, so we can prove the durable wrapper
records/replays and that ``create_agent`` accepts it and tool-calling flows.
"""

import asyncio

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool

import flyteplugins.agents.langchain._durable as durable_mod
from flyteplugins.agents.langchain._durable import DurableChatModel


class _FakeInner(BaseChatModel):
    """A canned chat model: emits a tool call first, then a final answer."""

    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-inner"

    def bind_tools(self, tools, **kwargs):
        # Emulate a real tool-calling model: format tool names and bind to self.
        formatted = [{"name": getattr(t, "name", str(t))} for t in tools]
        return self.bind(tools=formatted, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        object.__setattr__(self, "calls", self.calls + 1)
        has_tool_result = any(getattr(m, "type", None) == "tool" for m in messages)
        if not has_tool_result:
            msg = AIMessage(
                content="",
                tool_calls=[{"name": "get_weather", "args": {"city": "SF"}, "id": "call_1"}],
            )
        else:
            msg = AIMessage(content="It is sunny in SF")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return asyncio.get_event_loop().run_until_complete(self._agenerate(messages, stop, None, **kwargs))


def _weather_tool():
    async def get_weather(city: str) -> str:
        return f"sunny in {city}"

    return StructuredTool.from_function(coroutine=get_weather, name="get_weather", description="get weather")


def test_result_serialization_round_trip():
    """A ChatResult's messages survive dumps -> loads unchanged."""
    original = ChatResult(
        generations=[
            ChatGeneration(
                message=AIMessage(
                    content="",
                    tool_calls=[{"name": "get_weather", "args": {"city": "SF"}, "id": "call_1"}],
                )
            )
        ]
    )
    rebuilt = durable_mod._loads_result(durable_mod._dumps_result(original))
    assert rebuilt.generations[0].message.content == ""
    assert rebuilt.generations[0].message.tool_calls[0]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_wrapper_delegates_to_inner_once():
    """Outside a task, durable_step is a pass-through: inner is called exactly once."""
    inner = _FakeInner()
    dm = DurableChatModel(inner=inner)
    result = await dm._agenerate([])
    assert inner.calls == 1
    assert result.generations[0].message.tool_calls[0]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_durable_step_records_and_replays(monkeypatch):
    """Stub durable_step to assert the wrapper records the turn and replays it via loads."""
    recorded = {}

    async def fake_durable_step(key, run, *, name, dumps, loads):
        # First call: run the real work and record its serialized form.
        if key not in recorded:
            recorded[key] = dumps(await run())
        # Replay path: rebuild from the recorded payload without calling run again.
        return loads(recorded[key])

    monkeypatch.setattr(durable_mod, "durable_step", fake_durable_step)

    inner = _FakeInner()
    dm = DurableChatModel(inner=inner)

    first = await dm._agenerate([])
    assert inner.calls == 1
    assert first.generations[0].message.tool_calls[0]["name"] == "get_weather"

    # A second identical turn replays from the record — inner is NOT called again.
    second = await dm._agenerate([])
    assert inner.calls == 1
    assert second.generations[0].message.tool_calls[0]["name"] == "get_weather"


def test_bind_tools_routes_generation_through_wrapper():
    """bind_tools binds the formatted tool kwargs to the wrapper, not the inner model."""
    inner = _FakeInner()
    dm = DurableChatModel(inner=inner)
    bound = dm.bind_tools([_weather_tool()])
    # The runnable is bound to the durable wrapper so generation routes through it.
    assert bound.bound is dm
    assert "tools" in bound.kwargs


@pytest.mark.asyncio
async def test_end_to_end_create_agent_with_durable_model():
    """Build DurableChatModel over a fake inner, drive create_agent, assert final text."""
    try:
        from langchain.agents import create_agent
    except Exception:
        pytest.skip("langchain.agents.create_agent not available")

    inner = _FakeInner()
    dm = DurableChatModel(inner=inner)
    graph = create_agent(dm, [_weather_tool()], system_prompt="You are helpful.")

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "weather in SF?"}]})
    assert result["messages"][-1].content == "It is sunny in SF"
    # Two model turns: the tool-call turn and the final-answer turn.
    assert inner.calls == 2
