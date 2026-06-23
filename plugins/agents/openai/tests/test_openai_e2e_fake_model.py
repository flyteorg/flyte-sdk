"""End-to-end loop test with a fake model — no network, no API key.

Drives ``run_agent`` through a full turn → tool call → final answer cycle using a
stub ``Model``, proving the tool bridge dispatches the Flyte task and the durable
model provider wraps the loop transparently (outside a task context, the
``flyte.trace`` recording is a pass-through).
"""

import flyte
import pytest

# The fake model builds OpenAI Responses output items directly; if a given SDK
# version exposes a different shape, skip rather than fail the suite.
agents = pytest.importorskip("agents")
try:
    from agents.items import ModelResponse
    from agents.models.interface import Model, ModelProvider
    from agents.usage import Usage
    from openai.types.responses import (
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputText,
    )
except Exception:  # pragma: no cover - shape drift across SDK versions
    pytest.skip("openai-agents response types not available in this version", allow_module_level=True)

from agents import RunConfig  # noqa: E402

from flyteplugins.agents.openai import function_tool, run_agent  # noqa: E402

_env = flyte.TaskEnvironment("agents_e2e")


def _final(text: str) -> "ModelResponse":
    return ModelResponse(
        output=[
            ResponseOutputMessage(
                id="m1",
                type="message",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
            )
        ],
        usage=Usage(),
        response_id=None,
    )


def _tool_call(name: str, arguments: str) -> "ModelResponse":
    return ModelResponse(
        output=[ResponseFunctionToolCall(type="function_call", call_id="c1", name=name, arguments=arguments)],
        usage=Usage(),
        response_id=None,
    )


@pytest.mark.asyncio
async def test_run_agent_full_loop_with_tool_call():
    executed = {"count": 0}

    @function_tool
    @_env.task
    async def get_weather(city: str) -> str:
        """Get weather."""
        executed["count"] += 1
        return f"sunny in {city}"

    class _FakeModel(Model):
        def __init__(self):
            self.turns = 0

        async def get_response(self, *args, **kwargs):
            self.turns += 1
            # First turn asks for the tool; second turn answers.
            return _tool_call("get_weather", '{"city": "Paris"}') if self.turns == 1 else _final("Sunny in Paris.")

        def stream_response(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    fake = _FakeModel()

    class _FakeProvider(ModelProvider):
        def get_model(self, name):
            return fake

    out = await run_agent(
        "weather in Paris?",
        tools=[get_weather],
        durable=True,
        observability=False,
        run_config=RunConfig(model_provider=_FakeProvider()),
    )

    assert "sunny in paris" in out.lower()
    assert fake.turns == 2
    assert executed["count"] == 1
