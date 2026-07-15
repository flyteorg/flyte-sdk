"""Tests for FlyteAgent — build your own agent with Flyte durability."""

import sys

import langchain.agents  # noqa: F401
import pytest

import flyteplugins.agents.langchain._flyte_agent  # noqa: F401
import flyteplugins.agents.langchain._flyte_agent as flyte_agent_mod


class _FakeResult:
    """A minimal fake LangChain agent result."""

    def __init__(self, data):
        self._data = data

    def get(self, key, default=""):
        if key == "output":
            return self._data
        return default


class _FakeAgent:
    """A minimal fake LangChain AgentExecutor."""

    def __init__(self, result):
        self._result = result

    def __call__(self, *args, **kwargs):
        return self

    async def ainvoke(self, input_dict, **kwargs):
        return self._result


@pytest.mark.asyncio
async def test_flyte_agent_build(monkeypatch):
    """FlyteAgent.build returns a callable agent with ainvoke when tools are provided."""
    fake_agent = _FakeAgent(_FakeResult("The weather is sunny."))

    # Mock AgentExecutor at the langchain.agents level so the import succeeds.
    # langchain.agents uses __getattr__ that raises AttributeError for missing
    # attributes, so monkeypatch.setattr cannot add new ones. Patch via
    # sys.modules __dict__ directly and restore originals on cleanup.
    class _FakeAgentExecutor:
        def __call__(self, *a, **k):
            return fake_agent

    lc_agents_mod = sys.modules["langchain.agents"]
    _orig_ae = lc_agents_mod.__dict__.get("AgentExecutor")
    _orig_cta = lc_agents_mod.__dict__.get("create_tool_calling_agent")
    lc_agents_mod.__dict__["AgentExecutor"] = _FakeAgentExecutor()
    lc_agents_mod.__dict__["create_tool_calling_agent"] = _FakeAgentExecutor()

    # Mock _ChatOpenAI at module level to avoid requiring API credentials.
    monkeypatch.setattr(
        flyte_agent_mod, "_ChatOpenAI", type("_FakeChatOpenAI", (), {"__init__": lambda self, *a, **k: None})
    )

    from flyte import TaskEnvironment

    from flyteplugins.agents.langchain import FlyteAgent

    env = TaskEnvironment("fa_build_a")

    @env.task
    async def get_weather(city: str) -> str:
        return f"sunny in {city}"

    agent = FlyteAgent(name="test-agent")
    tools = agent.durable_tools(get_weather)
    built = agent.build(tools=tools)

    assert built is not None
    assert hasattr(built, "ainvoke")

    # Restore original module attributes
    if _orig_ae is not None:
        lc_agents_mod.__dict__["AgentExecutor"] = _orig_ae
    else:
        lc_agents_mod.__dict__.pop("AgentExecutor", None)
    if _orig_cta is not None:
        lc_agents_mod.__dict__["create_tool_calling_agent"] = _orig_cta
    else:
        lc_agents_mod.__dict__.pop("create_tool_calling_agent", None)


@pytest.mark.asyncio
async def test_flyte_agent_durable_tools():
    """FlyteAgent.durable_tools returns a list of durable tool wrappers."""
    from flyte import TaskEnvironment

    from flyteplugins.agents.langchain import FlyteAgent

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

    from flyteplugins.agents.langchain import FlyteAgent

    flyte_agent = FlyteAgent()
    built = flyte_agent.build(agent=fake_agent)

    assert built is fake_agent


@pytest.mark.asyncio
async def test_flyte_agent_properties():
    """FlyteAgent exposes correct properties."""
    from flyteplugins.agents.langchain import FlyteAgent

    agent = FlyteAgent(name="my-agent", instructions="Be concise.")

    assert agent.name == "my-agent"
    assert agent.instructions == "Be concise."

    agent2 = FlyteAgent(name="default-instructions")
    assert "default-instructions" in agent2.instructions
