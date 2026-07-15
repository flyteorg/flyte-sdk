"""Tests for FlyteAgent — build your own agent with Flyte durability."""

import pytest


class _FakeResult:
    """A minimal fake LangGraph agent result."""

    def __init__(self, data):
        self._data = data

    def get(self, key, default=None):
        if default is None:
            return self._data if key == "__all__" else None
        return self._data if key == "__all__" else default


class _FakeAgent:
    """A minimal fake compiled LangGraph StateGraph."""

    def __init__(self, result):
        self._result = result

    async def ainvoke(self, input_dict, **kwargs):
        return self._result


@pytest.mark.asyncio
async def test_flyte_agent_build():
    """FlyteAgent.build returns a LangGraph StateGraph when tools are provided."""
    from flyte import TaskEnvironment

    from flyteplugins.agents.langgraph import FlyteAgent

    env = TaskEnvironment("fa_build_a")

    @env.task
    async def get_weather(city: str) -> str:
        return f"sunny in {city}"

    agent = FlyteAgent(name="test-agent")
    tools = agent.durable_tools(get_weather)
    built = agent.build(tools=tools)

    assert built is not None
    assert hasattr(built, "ainvoke")


@pytest.mark.asyncio
async def test_flyte_agent_durable_tools():
    """FlyteAgent.durable_tools returns a list of durable tool wrappers."""
    from flyte import TaskEnvironment

    from flyteplugins.agents.langgraph import FlyteAgent

    env = TaskEnvironment("fa_tools_a")

    @env.task
    async def get_weather(city: str) -> str:
        return f"sunny in {city}"

    @env.task
    async def get_news(topic: str) -> str:
        return f"news about {topic}"

    flyte_agent = FlyteAgent()
    tools = flyte_agent.durable_tools(get_weather, get_news)

    assert len(tools) == 2
    # Each tool should be callable
    assert callable(tools[0])


@pytest.mark.asyncio
async def test_flyte_agent_with_prebuilt_agent():
    """FlyteAgent.build returns a pre-built agent when provided."""
    fake_agent = _FakeAgent(_FakeResult("Hello!"))

    from flyteplugins.agents.langgraph import FlyteAgent

    flyte_agent = FlyteAgent()
    built = flyte_agent.build(agent=fake_agent)

    assert built is fake_agent


@pytest.mark.asyncio
async def test_flyte_agent_properties():
    """FlyteAgent exposes correct properties."""
    from flyteplugins.agents.langgraph import FlyteAgent

    agent = FlyteAgent(name="my-agent", instructions="Be concise.")

    assert agent.name == "my-agent"
    assert agent.instructions == "Be concise."

    agent2 = FlyteAgent(name="default-instructions")
    assert "default-instructions" in agent2.instructions
