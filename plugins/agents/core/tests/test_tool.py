"""Tests for the shared generic ``tool`` (plain-callable tool wrapper).

This lives in core because the Mistral and Google ADK adapters share it verbatim
(their SDKs accept plain Python callables as tools). OpenAI/Claude provide their own
SDK-native versions, so they are exercised in their own packages.
"""

import inspect
import json
from unittest.mock import AsyncMock, patch

import flyte
import pytest

from flyteplugins.agents.core import ToolTaskResolver, coerce_tool_args, tool, task_json_schema


def test_task_becomes_plain_tool_with_resolver():
    env = flyte.TaskEnvironment("core_ft_a")

    @tool
    @env.task
    def get_weather(city: str) -> str:
        """Get weather."""
        return city

    assert inspect.isfunction(get_weather)
    assert get_weather.__name__ == "get_weather"
    assert get_weather.__wrapped_task__ is get_weather.task
    assert isinstance(get_weather.task.task_resolver, ToolTaskResolver)


def test_wrapper_preserves_the_task_signature():
    env = flyte.TaskEnvironment("core_ft_b")

    @tool
    @env.task
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    # functools.wraps -> the SDK derives the right declaration from the real params.
    assert list(inspect.signature(add).parameters) == ["a", "b"]


def test_plain_callable_passes_through():
    def f(x: int) -> int:
        return x

    assert tool(f) is f


def test_callable_class_instance_is_supported():
    # An instance of a class defining __call__ is a valid tool: SDKs that accept
    # plain callables inspect it through its __call__.
    class Search:
        def __call__(self, query: str) -> str:
            """Search the web."""
            return query

    s = Search()
    assert tool(s) is s


def test_name_and_description_override_a_plain_callable():
    def f(x: int) -> int:
        return x

    out = tool(f, name="renamed", description="Overridden.")
    assert out is f
    assert out.__name__ == "renamed"
    assert out.__doc__ == "Overridden."


def test_non_callable_is_rejected():
    with pytest.raises(TypeError):
        tool(42)


@pytest.mark.asyncio
async def test_tool_dispatches_to_task_aio():
    env = flyte.TaskEnvironment("core_ft_c")

    @tool
    @env.task
    def multiply(a: int, b: int) -> int:
        """Multiply."""
        return a * b

    with patch.object(multiply.task, "aio", new_callable=AsyncMock, return_value=42) as mock_aio:
        result = await multiply(a=6, b=7)

    mock_aio.assert_awaited_once_with(a=6, b=7)
    assert result == 42


def test_coerce_tool_args_int_to_float():
    # LLMs emit JSON numbers without a decimal as int; Flyte's type engine rejects
    # int for a float param, so we coerce it before dispatch.
    env = flyte.TaskEnvironment("core_ft_coerce")

    @env.task
    def issue_refund(account_id: str, amount_usd: float) -> str:
        return "ok"

    out = coerce_tool_args(issue_refund, {"account_id": "A-1", "amount_usd": 42})
    assert out == {"account_id": "A-1", "amount_usd": 42.0}
    assert isinstance(out["amount_usd"], float)


def test_coerce_tool_args_leaves_ints_and_bools_alone():
    env = flyte.TaskEnvironment("core_ft_coerce2")

    @env.task
    def f(n: int, flag: bool, ratio: float) -> str:
        return "ok"

    out = coerce_tool_args(f, {"n": 5, "flag": True, "ratio": 2})
    assert out["n"] == 5 and isinstance(out["n"], int)  # int param untouched
    assert out["flag"] is True  # bool not coerced to float
    assert out["ratio"] == 2.0 and isinstance(out["ratio"], float)  # float param coerced


@pytest.mark.asyncio
async def test_dispatch_coerces_int_to_float_for_a_float_param():
    env = flyte.TaskEnvironment("core_ft_coerce3")

    @tool
    @env.task
    def refund(account_id: str, amount_usd: float) -> str:
        """Refund."""
        return "ok"

    with patch.object(refund.task, "aio", new_callable=AsyncMock, return_value="ok") as mock_aio:
        await refund(account_id="A-1", amount_usd=42)  # int from the "model"

    mock_aio.assert_awaited_once_with(account_id="A-1", amount_usd=42.0)


def test_task_json_schema_describes_the_task_inputs():
    # Adapters that want a JSON-schema tool definition derive it from the task via the
    # Flyte type engine; the schema must name the task's parameters.
    env = flyte.TaskEnvironment("core_ft_schema")

    @env.task
    def lookup(account_id: str, amount: float) -> str:
        """Look up an account balance."""
        return "ok"

    schema = task_json_schema(lookup)
    assert isinstance(schema, dict)
    blob = json.dumps(schema)
    assert "account_id" in blob and "amount" in blob
