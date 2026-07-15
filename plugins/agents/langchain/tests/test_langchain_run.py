"""Tests for LangChain run_agent (mocked)."""

from unittest.mock import MagicMock

import flyte
import pytest

import flyteplugins.agents.langchain._run as run_mod


class _FakeAgent:
    """A minimal fake LangChain agent executor."""

    def __init__(self, result):
        self._result = result

    async def ainvoke(self, input_dict, **kwargs):
        return self._result


@pytest.mark.asyncio
async def test_run_agent_with_tools(monkeypatch):
    """run_agent builds an agent from tools and runs it."""
    env = flyte.TaskEnvironment("lc_run_a")

    @env.task
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    fake_agent = _FakeAgent({"output": "The weather is sunny."})

    def fake_build(*args, **kwargs):
        return fake_agent

    mock_chat_openai = MagicMock(spec=[])
    # Patch _ChatOpenAI so the real ChatOpenAI import is skipped entirely.
    monkeypatch.setattr(run_mod, "_ChatOpenAI", mock_chat_openai)
    monkeypatch.setattr(run_mod, "_AgentExecutor", fake_build)

    result = await run_mod.run_agent("What's the weather?", tools=[get_weather], name="test-agent")
    assert result is not None


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
