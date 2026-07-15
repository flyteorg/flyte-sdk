"""Tests for FlyteAgent — build your own agent with Flyte durability."""

import pytest


class _FakeResult:
    """A minimal fake Pydantic AI run result."""

    def __init__(self, data):
        self.data = data


class _FakeAgent:
    """A minimal fake Pydantic AI agent."""

    def __init__(self, result):
        self._result = result

    def __call__(self, *args, **kwargs):
        return self

    async def run(self, message, **kwargs):
        return self._result


@pytest.mark.asyncio
async def test_flyte_agent_build(monkeypatch):
    """FlyteAgent.build returns a pydantic_ai.Agent when tools are provided."""
    import flyteplugins.agents.pydantic_ai._flyte_agent as flyte_agent_mod

    fake_agent = _FakeAgent(_FakeResult("built"))

    # Mock pydantic_ai.Agent before FlyteAgent is imported
    monkeypatch.setattr(flyte_agent_mod, "_PydanticAgent", fake_agent)

    from flyte import TaskEnvironment

    from flyteplugins.agents.pydantic_ai import FlyteAgent

    env = TaskEnvironment("fa_build_a")

    @env.task
    async def get_weather(city: str) -> str:
        return f"sunny in {city}"

    agent = FlyteAgent("gpt-4o", name="test-agent")
    tools = agent.durable_tools(get_weather)
    built = agent.build(tools=tools)

    assert built is not None
    assert hasattr(built, "run")


@pytest.mark.asyncio
async def test_flyte_agent_durable_tools():
    """FlyteAgent.durable_tools returns a list of durable tool wrappers."""
    from flyte import TaskEnvironment

    from flyteplugins.agents.pydantic_ai import FlyteAgent

    env = TaskEnvironment("fa_tools_a")

    @env.task
    async def get_weather(city: str) -> str:
        return f"sunny in {city}"

    @env.task
    async def get_news(topic: str) -> str:
        return f"news about {topic}"

    flyte_agent = FlyteAgent("gpt-4o")
    tools = flyte_agent.durable_tools(get_weather, get_news)

    assert len(tools) == 2
    # Each tool should be callable
    assert callable(tools[0])


@pytest.mark.asyncio
async def test_flyte_agent_run(monkeypatch):
    """FlyteAgent.run builds and runs an agent, returning the final answer."""
    from flyte import TaskEnvironment

    import flyteplugins.agents.pydantic_ai._flyte_agent as flyte_agent_mod
    from flyteplugins.agents.pydantic_ai import FlyteAgent

    env = TaskEnvironment("fa_run_a")

    @env.task
    async def get_weather(city: str) -> str:
        return f"sunny in {city}"

    fake_agent = _FakeAgent(_FakeResult("The weather is sunny."))

    # Mock pydantic_ai.Agent before FlyteAgent is imported
    monkeypatch.setattr(flyte_agent_mod, "_PydanticAgent", fake_agent)

    flyte_agent = FlyteAgent("gpt-4o", name="test-agent")
    result = await flyte_agent.run("What's the weather?", tools=[get_weather])

    assert result == "The weather is sunny."


@pytest.mark.asyncio
async def test_flyte_agent_with_prebuilt_agent():
    """FlyteAgent.build returns a pre-built agent when provided."""
    fake_agent = _FakeAgent(_FakeResult("Hello!"))

    from flyteplugins.agents.pydantic_ai import FlyteAgent

    flyte_agent = FlyteAgent("gpt-4o")
    built = flyte_agent.build(agent=fake_agent)

    assert built is fake_agent


@pytest.mark.asyncio
async def test_flyte_agent_properties():
    """FlyteAgent exposes correct properties."""
    from flyteplugins.agents.pydantic_ai import FlyteAgent

    agent = FlyteAgent("gpt-4o", name="my-agent", system_prompt="Be concise.")

    assert agent.name == "my-agent"
    assert agent.system_prompt == "Be concise."

    agent2 = FlyteAgent("gpt-4o", name="default-prompt")
    assert "default-prompt" in agent2.system_prompt
