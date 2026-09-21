"""Tests for the Hermes report timeline (mocked — no network).

The adapter renders from two seams it already owns: the `llm_execution`
middleware chain (model turns) and the tool wrapper in `_tools` (tool calls).
These drive both through a fake agent and assert on the rows that reach the
`ReportTimeline`.
"""

import types

import flyte
import pytest

import flyteplugins.agents.hermes._durable as durable_mod
import flyteplugins.agents.hermes._observability as obs_mod
import flyteplugins.agents.hermes._run as run_mod
from flyteplugins.agents.hermes import tool


class _RecordingTimeline:
    """Stands in for `ReportTimeline`, keeping the rows instead of rendering them."""

    def __init__(self, *args, **kwargs):
        self.headings = []
        self.rows = []

    def heading(self, text):
        self.headings.append(str(text))

    def row(self, **kwargs):
        self.rows.append(kwargs)


@pytest.fixture
def timeline(monkeypatch):
    """Make `run_agent` build a recording timeline instead of a report-backed one."""
    captured = _RecordingTimeline()
    monkeypatch.setattr(run_mod, "ReportTimeline", lambda *a, **k: captured)
    return captured


@pytest.fixture
def clean_middleware():
    """Restore the process-global Hermes middleware list after the test."""
    from hermes_cli.plugins import get_plugin_manager

    callbacks = get_plugin_manager()._middleware.setdefault("llm_execution", [])
    before = list(callbacks)
    try:
        yield callbacks
    finally:
        callbacks[:] = before


class _ToolCallingAgent:
    """A fake agent that takes one model turn through the real middleware chain and calls a tool."""

    def __init__(self, tool_name):
        self._tool_name = tool_name

    def run_conversation(self, user_message, **kwargs):
        from hermes_cli.middleware import run_llm_execution_middleware
        from tools.registry import registry

        def _perform_api_call(request):
            return types.SimpleNamespace(
                model="test-model",
                usage=types.SimpleNamespace(prompt_tokens=11, completion_tokens=5, total_tokens=16),
                choices=[
                    types.SimpleNamespace(
                        finish_reason="tool_calls",
                        message=types.SimpleNamespace(
                            role="assistant",
                            content="thinking",
                            tool_calls=[
                                types.SimpleNamespace(
                                    function=types.SimpleNamespace(name=self._tool_name, arguments="{}")
                                )
                            ],
                        ),
                    )
                ],
            )

        run_llm_execution_middleware(
            {"model": "test-model", "messages": [{"role": "user", "content": user_message}]},
            _perform_api_call,
            api_mode="chat_completions",
            api_call_count=1,
        )

        # The real dispatch path: the registry bridges the async handler itself.
        registry.dispatch(self._tool_name, {"city": "Paris"})
        return {"final_response": "sunny in Paris"}


@pytest.fixture
def weather_tool():
    """A Flyte task registered as a Hermes tool, under a name unique to this test."""
    env = flyte.TaskEnvironment("h_obs")

    @tool
    @env.task
    def obs_get_weather(city: str) -> str:
        """Get weather."""
        return f"sunny in {city}"

    return obs_get_weather


@pytest.mark.asyncio
async def test_timeline_records_model_turns_and_tool_calls(timeline, clean_middleware, weather_tool):
    """A run with observability on renders a row per model turn and per tool call."""
    agent = _ToolCallingAgent("obs_get_weather")
    assert await run_mod.run_agent("weather in Paris?", agent=agent, durable=False) == "sunny in Paris"

    assert timeline.headings == ["Hermes agent"]
    labels = [row["label"] for row in timeline.rows]
    assert labels == ["test-model", "obs_get_weather", "final answer"]

    turn = timeline.rows[0]
    assert "model turn" in turn["meta"] and "ms" in turn["meta"]
    assert "thinking" in turn["detail"]
    assert "obs_get_weather" in turn["detail"]
    assert "total_tokens=16" in turn["detail"]

    tool_row = timeline.rows[1]
    assert "tool" in tool_row["meta"] and "ms" in tool_row["meta"]
    assert "Paris" in tool_row["detail"]
    assert not tool_row["error"]

    # The recorder is closed again once the run finishes.
    assert obs_mod.current_recorder() is None


@pytest.mark.asyncio
async def test_observability_false_appends_nothing(timeline, clean_middleware, weather_tool):
    """observability=False is a true no-op: no heading, no rows, no recorder."""
    agent = _ToolCallingAgent("obs_get_weather")
    assert await run_mod.run_agent("weather?", agent=agent, durable=False, observability=False) == "sunny in Paris"

    assert timeline.headings == []
    assert timeline.rows == []
    assert obs_mod.current_recorder() is None


@pytest.mark.asyncio
async def test_observer_middleware_registers_once(timeline, clean_middleware):
    """Registration is idempotent, and the observer never replaces the response."""

    class _PlainAgent:
        def __init__(self):
            self.raw = None

        def run_conversation(self, user_message, **kwargs):
            from hermes_cli.middleware import run_llm_execution_middleware

            def _perform_api_call(request):
                self.raw = types.SimpleNamespace(
                    model="m",
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(content="hi", tool_calls=None),
                        )
                    ],
                )
                return self.raw

            self.seen = run_llm_execution_middleware(
                {"model": "m", "messages": [{"role": "user", "content": user_message}]},
                _perform_api_call,
            )
            return {"final_response": "hi"}

    agent = _PlainAgent()
    await run_mod.run_agent("hi", agent=agent, durable=False)
    await run_mod.run_agent("hi", agent=agent, durable=False)

    registered = [cb for cb in clean_middleware if cb is obs_mod.observe_llm_execution]
    assert registered == [obs_mod.observe_llm_execution]
    # Pure observer: the agent sees the provider object itself.
    assert agent.seen is agent.raw


@pytest.mark.asyncio
async def test_durable_live_turn_renders_exactly_one_row(timeline, clean_middleware, weather_tool):
    """With both middlewares registered a live turn still yields a single model row."""
    agent = _ToolCallingAgent("obs_get_weather")
    assert await run_mod.run_agent("weather in Paris?", agent=agent, durable=True) == "sunny in Paris"

    labels = [row["label"] for row in timeline.rows]
    assert labels == ["test-model", "obs_get_weather", "final answer"]

    turn = timeline.rows[0]
    assert turn["icon"] == obs_mod._MODEL_ICON
    assert "replayed" not in turn["meta"]
    assert "ms" in turn["meta"]
    assert "total_tokens=16" in turn["detail"]


@pytest.mark.asyncio
async def test_replayed_turn_is_rendered_and_marked(timeline, clean_middleware, monkeypatch):
    """A turn answered from its trace record gets a row, marked replayed, with no call."""
    recorded = durable_mod._dumps(
        types.SimpleNamespace(
            model="test-model",
            usage=types.SimpleNamespace(total_tokens=16),
            choices=[
                types.SimpleNamespace(
                    finish_reason="stop",
                    message=types.SimpleNamespace(role="assistant", content="sunny in Paris", tool_calls=None),
                )
            ],
        )
    )

    async def _replay(request_key, run, *, name="durable_step", dumps=None, loads=None):
        """Stand in for `durable_step` on a retried attempt: never run the step."""
        return loads(recorded)

    monkeypatch.setattr(durable_mod, "durable_step", _replay)

    class _ReplayingAgent:
        """Drives one turn through the real chain; `next_call` must never run."""

        def __init__(self):
            self.calls = 0

        def run_conversation(self, user_message, **kwargs):
            from hermes_cli.middleware import run_llm_execution_middleware

            def _perform_api_call(request):
                self.calls += 1
                raise AssertionError("the provider must not be called on a replayed turn")

            self.seen = run_llm_execution_middleware(
                {"model": "test-model", "messages": [{"role": "user", "content": user_message}]},
                _perform_api_call,
                api_mode="chat_completions",
                api_call_count=1,
            )
            return {"final_response": "sunny in Paris"}

    agent = _ReplayingAgent()
    assert await run_mod.run_agent("weather in Paris?", agent=agent, durable=True) == "sunny in Paris"

    assert agent.calls == 0
    assert agent.seen.choices[0].message.content == "sunny in Paris"

    labels = [row["label"] for row in timeline.rows]
    assert labels == ["test-model", "final answer"]

    turn = timeline.rows[0]
    assert turn["icon"] == obs_mod._REPLAY_ICON
    # Marked replayed, and with no duration: the message count is all that follows.
    assert turn["meta"] == "model turn · replayed · 1 msgs"
    assert "total_tokens" not in turn["detail"]
    assert "sunny in Paris" in turn["detail"]


def test_rendering_failure_never_breaks_the_run():
    """A timeline that raises is swallowed; the recorder still counts the event."""

    class _BrokenTimeline:
        def row(self, **kwargs):
            raise RuntimeError("boom")

    recorder = obs_mod._RunRecorder(_BrokenTimeline())
    recorder.model_turn({"model": "m", "messages": []}, None, 0.01)
    recorder.tool_call("t", {"a": 1}, "out", 0.01)
    assert (recorder.turns, recorder.tool_calls) == (1, 1)


def test_record_tool_call_is_a_noop_without_a_recorder():
    assert obs_mod.current_recorder() is None
    obs_mod.record_tool_call("t", {}, "out", 0.0)  # must not raise


def test_response_text_and_calls_reads_the_anthropic_shape():
    response = types.SimpleNamespace(
        content=[
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "name": "get_weather"},
        ]
    )
    assert obs_mod._response_text_and_calls(response) == ("hello", ["get_weather"])
