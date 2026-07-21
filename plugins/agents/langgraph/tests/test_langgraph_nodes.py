"""Unit tests for the LangGraph node factories (no network / no controller)."""

from unittest.mock import AsyncMock, patch

import flyte
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from flyteplugins.agents.langgraph import ai_node, tool, tool_node


class _FakeBound:
    def __init__(self, response):
        self._response = response

    async def ainvoke(self, messages):
        return self._response


class _FakeModel:
    def __init__(self, response):
        self._response = response
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return _FakeBound(self._response)


@pytest.mark.asyncio
async def test_ai_node_binds_tools_and_appends_response():
    env = flyte.TaskEnvironment("lg_nodes_a")

    @tool
    @env.task
    def get_weather(city: str) -> str:
        """Weather."""
        return city

    response = AIMessage(content="hello")
    model = _FakeModel(response)
    node = ai_node(model, [get_weather], durable=False, observability=False)

    assert model.bound_tools == [get_weather]
    out = await node({"messages": [HumanMessage(content="hi")]})
    assert out["messages"] == [response]


@pytest.mark.asyncio
async def test_ai_node_durable_round_trips_response():
    """durable=True serializes/rebuilds the model turn (outside a task, trace is
    a pass-through, so the dumps/loads round-trip is still exercised)."""
    response = AIMessage(content="hello")
    node = ai_node(_FakeModel(response), [], durable=True, observability=False)
    out = await node({"messages": [HumanMessage(content="hi")]})
    (msg,) = out["messages"]
    assert type(msg).__name__ == "AIMessage"
    assert msg.content == "hello"


@pytest.mark.asyncio
async def test_tool_node_runs_tool_calls_durably():
    env = flyte.TaskEnvironment("lg_nodes_b")

    @tool
    @env.task
    def get_weather(city: str) -> str:
        """Weather."""
        return f"sunny in {city}"

    node = tool_node([get_weather], observability=False)

    ai = AIMessage(
        content="",
        tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}, "id": "call_1"}],
    )
    with patch.object(get_weather.task, "aio", new_callable=AsyncMock, return_value="sunny in Paris") as mock_aio:
        out = await node({"messages": [ai]})

    mock_aio.assert_awaited_once_with(city="Paris")
    (msg,) = out["messages"]
    assert msg.content == "sunny in Paris"
    assert msg.tool_call_id == "call_1"


@pytest.mark.asyncio
async def test_tool_node_reports_unknown_tool():
    node = tool_node([], observability=False)
    ai = AIMessage(content="", tool_calls=[{"name": "nope", "args": {}, "id": "x"}])
    out = await node({"messages": [ai]})
    assert "unknown tool" in out["messages"][0].content
