"""Unit tests for the LangChain adapter's tool bridge (no network / no controller)."""

from unittest.mock import AsyncMock, patch

import flyte
import pytest
from flyteplugins.agents.core import ToolTaskResolver

from flyteplugins.agents.langchain import tool


def test_task_becomes_langchain_tool_with_resolver():
    env = flyte.TaskEnvironment("lc_tools_a")

    @tool
    @env.task
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"sunny in {city}"

    # The StructuredTool carries the task + resolver. (It is a pydantic model, so
    # the tool's public name is ``.name``, not ``__name__``.)
    assert get_weather.name == "get_weather"
    assert get_weather.__wrapped_task__ is get_weather.task
    assert isinstance(get_weather.task.task_resolver, ToolTaskResolver)


def test_langchain_tool_is_a_basetool_with_args_schema():
    """The wrapped tool is a real BaseTool with a schema derived from the task."""
    from langchain_core.tools import BaseTool

    env = flyte.TaskEnvironment("lc_tools_b")

    @tool
    @env.task
    def get_weather(city: str) -> str:
        """Get weather."""
        return city

    assert isinstance(get_weather, BaseTool)
    assert get_weather.name == "get_weather"
    # The args schema mirrors the task's typed signature (not a `kwargs` object).
    assert set(get_weather.args) == {"city"}
    assert get_weather.args["city"]["type"] == "string"


def test_langchain_agent_accepts_the_tool():
    """A wrapped tool drops into create_agent (compiled graph exposes ainvoke)."""
    try:
        from langchain.agents import create_agent
    except Exception:
        pytest.skip("langchain.agents.create_agent not available")

    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    class _ToolCapableFake(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    env = flyte.TaskEnvironment("lc_tools_b2")

    @tool
    @env.task
    def get_weather(city: str) -> str:
        """Get weather."""
        return city

    model = _ToolCapableFake(messages=iter([AIMessage(content="ok")]))
    graph = create_agent(model, [get_weather], system_prompt="You are helpful.")
    assert graph is not None
    assert hasattr(graph, "ainvoke")


@pytest.mark.asyncio
async def test_tool_dispatches_to_task_aio():
    env = flyte.TaskEnvironment("lc_tools_c")

    @tool
    @env.task
    def multiply(a: int, b: int) -> int:
        """Multiply."""
        return a * b

    # The StructuredTool is invoked via its async run interface, which dispatches to
    # the backing task's ``aio`` (a durable child action).
    with patch.object(multiply.task, "aio", new_callable=AsyncMock, return_value=42) as mock_aio:
        result = await multiply.ainvoke({"a": 6, "b": 7})

    mock_aio.assert_awaited_once_with(a=6, b=7)
    assert result == "42"


def test_bare_and_parametrized_decorator_forms():
    env = flyte.TaskEnvironment("lc_tools_d")

    @tool
    @env.task
    def a(x: int) -> int:
        """A."""
        return x

    assert a.name == "a"

    @tool(name="bee")
    @env.task
    def b(x: int) -> int:
        """B."""
        return x

    assert b.name == "bee"
