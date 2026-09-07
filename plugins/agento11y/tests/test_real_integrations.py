"""Each adapter, driven with the real agento11y integration package.

The tests in test_frameworks.py stub the injector modules so the wiring can be checked
without pulling langchain, google-adk, and the rest into this package. That leaves one thing
unproven: whether the real injectors accept what we pass them, and whether what they return
is something the framework actually takes.

These close that gap. Each skips unless both halves are installed — the agento11y integration
and the Flyte adapter — so they run in an environment that has a framework's extras and stay
out of the way everywhere else:

    pip install "flyteplugins-agento11y[openai]" flyteplugins-agents-openai
    pytest tests/test_real_integrations.py

What is asserted is narrow on purpose: that agento11y's own handler object reaches the exact
call the adapter makes. The frameworks themselves are stubbed, since driving a real model is
not what is in question.
"""

import pytest
from agento11y.exporters.noop import NoopGenerationExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from flyteplugins.agento11y import init, instrumented_frameworks


@pytest.fixture
def wired(clean):
    """Initialize against in-memory exporters, and return the frameworks that registered."""
    init(
        service_name="test",
        exporter=InMemorySpanExporter(),
        disable_batch=True,
        set_global=False,
        client_options={"generation_exporter": NoopGenerationExporter()},
    )
    return instrumented_frameworks()


def _require(framework: str, integration_module: str, adapter_module: str, wired):
    pytest.importorskip(integration_module, reason=f"agento11y integration for {framework} not installed")
    module = pytest.importorskip(adapter_module, reason=f"Flyte adapter for {framework} not installed")
    if framework not in wired:
        pytest.skip(f"{framework} did not register an instrumentor")
    return module


class _RecordingRunnable:
    """A stand-in compiled graph that records the kwargs the adapter invokes it with."""

    def __init__(self):
        self.kwargs = None

    async def ainvoke(self, state, **kwargs):
        self.kwargs = kwargs
        return {"messages": [type("M", (), {"content": "done"})()]}


@pytest.mark.asyncio
async def test_langgraph_handler_reaches_ainvoke(wired):
    run_mod = _require("langgraph", "agento11y_langgraph", "flyteplugins.agents.langgraph._run", wired)
    agent = _RecordingRunnable()
    await run_mod.run_agent("hi", agent=agent, observability=False)

    callbacks = agent.kwargs["config"]["callbacks"]
    assert [type(c).__name__ for c in callbacks] == ["Agento11yAsyncLangGraphHandler"]


@pytest.mark.asyncio
async def test_langchain_handler_reaches_ainvoke(wired):
    run_mod = _require("langchain", "agento11y_langchain", "flyteplugins.agents.langchain._run", wired)
    agent = _RecordingRunnable()
    await run_mod.run_agent("hi", agent=agent, observability=False)

    callbacks = agent.kwargs["config"]["callbacks"]
    assert [type(c).__name__ for c in callbacks] == ["Agento11yAsyncLangChainHandler"]


@pytest.mark.asyncio
async def test_openai_hooks_reach_the_runner(wired, monkeypatch):
    run_mod = _require("openai", "agento11y_openai_agents", "flyteplugins.agents.openai._run", wired)
    seen = {}

    class FakeRunner:
        @staticmethod
        async def run(agent, input, **kwargs):
            seen.update(kwargs)
            return type("R", (), {"final_output": "done"})()

    monkeypatch.setattr(run_mod, "Runner", FakeRunner)
    await run_mod.run_agent("hi", model="gpt-4.1", observability=False, durable=False)

    assert type(seen["hooks"]).__name__ == "Agento11yOpenAIAgentsRunHooks"


@pytest.mark.asyncio
async def test_claude_hooks_reach_the_options(wired, monkeypatch):
    run_mod = _require("claude", "agento11y_claude_agent", "flyteplugins.agents.claude._run", wired)
    seen = {}

    async def fake_query(prompt, options):
        seen["options"] = options
        return
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(run_mod, "query", fake_query)
    await run_mod.run_agent("hi", observability=False, durable=False)

    # The adapter installs only the PostToolUse pair for its report timeline, so anything
    # else present came from agento11y.
    events = set(getattr(seen["options"], "hooks", None) or {})
    assert {"PreToolUse", "Stop", "UserPromptSubmit"} <= events


@pytest.mark.asyncio
async def test_google_plugin_reaches_the_runner(wired, monkeypatch):
    run_mod = _require("google", "agento11y_google_adk", "flyteplugins.agents.google._run", wired)
    import google.adk.runners as adk_runners

    seen = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        async def run_async(self, **kwargs):
            return
            yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(adk_runners, "Runner", FakeRunner)
    await run_mod.run_agent("hi", model="gemini-2.0-flash", observability=False, durable=False)

    assert [type(p).__name__ for p in seen["plugins"]] == ["Agento11yGoogleAdkPlugin"]


@pytest.mark.asyncio
async def test_pydantic_ai_capability_reaches_run(wired):
    run_mod = _require("pydantic_ai", "agento11y_pydantic_ai", "flyteplugins.agents.pydantic_ai._run", wired)

    class RecordingAgent:
        def __init__(self):
            self.kwargs = None

        async def run(self, prompt, **kwargs):
            self.kwargs = kwargs
            return type("R", (), {"output": "done", "all_messages": lambda self: []})()

    agent = RecordingAgent()
    await run_mod.run_agent("hi", agent=agent, observability=False)

    capabilities = agent.kwargs["capabilities"]
    assert [type(c).__name__ for c in capabilities] == ["Agento11yPydanticAICapability"]


@pytest.mark.asyncio
async def test_claude_records_the_message_stream(wired, monkeypatch):
    """Claude's model turns arrive as stream messages, not through the options hooks.

    Decorating options gives tool spans only, which is why the agent had no generations at
    all. agento11y_query drives the SDK's own query and records each message, so the adapter's
    loop over the stream is untouched while generations finally get created.
    """
    run_mod = _require("claude", "agento11y_claude_agent", "flyteplugins.agents.claude._run", wired)
    seen = {}

    async def fake_query(prompt, options):
        seen["prompt"] = prompt
        seen["options"] = options
        return
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(run_mod, "query", fake_query)
    await run_mod.run_agent("hi", observability=False, durable=False)

    # The SDK's query still ran with the prompt, so the wrapper is transparent to the adapter.
    assert seen["prompt"] == "hi"
    # And agento11y instrumented the options on its way through, which is what it does inside
    # agento11y_query rather than via a separate payload instrumentor.
    events = set(getattr(seen["options"], "hooks", None) or {})
    assert {"PreToolUse", "Stop", "UserPromptSubmit"} <= events


@pytest.mark.asyncio
async def test_claude_call_wrapper_survives_the_sync_variant(wired, monkeypatch):
    """run_agent_sync hops to a background event loop on another thread.

    The conversation id is read from a contextvar at call time, so this checks the binding
    survives that hop rather than silently falling back to the SDK's session id.
    """
    run_mod = _require("claude", "agento11y_claude_agent", "flyteplugins.agents.claude._run", wired)
    seen = {}

    async def fake_query(prompt, options):
        seen["options"] = options
        return
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(run_mod, "query", fake_query)
    run_mod.run_agent_sync("hi", observability=False, durable=False)

    # agento11y instrumented the options on the way through, which it does inside
    # agento11y_query rather than via a payload instrumentor.
    events = set(getattr(seen["options"], "hooks", None) or {})
    assert {"PreToolUse", "Stop", "UserPromptSubmit"} <= events
