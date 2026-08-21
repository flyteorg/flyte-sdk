"""Tests for ``run_agent`` — config layering, prompt composition, sessions, cleanup.

Offline: ``FakeHarness`` stands in for ``DeepSeekHarness``, capturing the config
it was built with and the prompt it was driven with, and returning a canned
``RunResult``. One test has the fake harness actually execute a published tool
shim, which is the real contract: tool calls must be served while the harness's
blocking ``run`` is in flight.
"""

import asyncio
import json
import pathlib
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import flyte
import pytest
from deepseek_harness import DeepSeekHarnessConfig, Notification, RunResult

from flyteplugins.agents.deepseek import TOOLS_DIRNAME, run_agent, run_agent_sync, tool

CALLS = []


class FakeHarness:
    """Mimics DeepSeekHarness: a context manager with a blocking ``run``."""

    # Set by a test to run extra work inside ``run`` (e.g. call a tool shim).
    during_run = None
    # Notifications the fake streams back to ``on_notification``.
    stream = ()
    result_text = "the answer"
    finish_reason = "completed"

    def __init__(self, config):
        self.config = config
        CALLS.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def run(self, prompt, session_id=None, on_notification=None):
        self.prompt = prompt
        self.session_id = session_id
        for notification in type(self).stream:
            on_notification(notification)
        extra = type(self).during_run
        self.during_run_result = extra(self) if extra else None
        return RunResult(
            session_id=session_id or "generated",
            final_response=type(self).result_text,
            finish_reason=type(self).finish_reason,
            events=[{"type": "assistant/message"}],
            notifications=[],
        )


@pytest.fixture(autouse=True)
def _reset_fake():
    CALLS.clear()
    FakeHarness.during_run = None
    FakeHarness.stream = ()
    FakeHarness.result_text = "the answer"
    FakeHarness.finish_reason = "completed"
    yield
    FakeHarness.during_run = None


def _patch_harness():
    return patch("flyteplugins.agents.deepseek._run.DeepSeekHarness", FakeHarness)


def _weather_tool(env_name):
    env = flyte.TaskEnvironment(env_name)

    @tool
    @env.task
    async def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"sunny in {city}"

    return get_weather


@pytest.mark.asyncio
async def test_returns_the_final_response():
    with _patch_harness():
        assert await run_agent("hello", durable=False, observability=False) == "the answer"


@pytest.mark.asyncio
async def test_adapter_owns_workspace_and_session_root():
    """The workspace is where shims are published; the session root is what we mirror."""
    with _patch_harness():
        await run_agent("hello", durable=False, observability=False)

    cfg = CALLS[0].config
    assert isinstance(cfg, DeepSeekHarnessConfig)
    assert pathlib.Path(cfg.cwd).is_absolute()
    assert cfg.session_root is not None


@pytest.mark.asyncio
async def test_model_provider_and_max_tokens_are_layered_on():
    with _patch_harness():
        await run_agent(
            "hello",
            model="deepseek-v4-flash",
            provider="deepseek-official",
            max_tokens=1024,
            durable=False,
            observability=False,
        )

    cfg = CALLS[0].config
    assert (cfg.model, cfg.provider, cfg.max_tokens) == ("deepseek-v4-flash", "deepseek-official", 1024)


@pytest.mark.asyncio
async def test_a_caller_supplied_config_is_preserved():
    """SDK-native configuration survives; only cwd/session_root are the adapter's."""
    base = DeepSeekHarnessConfig(cordis="/tmp/custom.cordis.yml", base_url="https://proxy.example", model="base-model")
    with _patch_harness():
        await run_agent("hello", config=base, model="override-model", durable=False, observability=False)

    cfg = CALLS[0].config
    assert cfg.cordis == "/tmp/custom.cordis.yml"
    assert cfg.base_url == "https://proxy.example"
    assert cfg.model == "override-model"


@pytest.mark.asyncio
async def test_extra_kwargs_reach_the_sdk_config():
    with _patch_harness():
        await run_agent("hello", api_key="sk-test", request_timeout_seconds=12.0, durable=False, observability=False)

    cfg = CALLS[0].config
    assert cfg.api_key == "sk-test"
    assert cfg.request_timeout_seconds == 12.0


@pytest.mark.asyncio
async def test_instructions_become_the_runtime_system_prompt():
    with _patch_harness():
        await run_agent("hello", instructions="Be concise.", durable=False, observability=False)

    assert CALLS[0].config.env["DSH_SYSTEM_PROMPT"] == "Be concise."


@pytest.mark.asyncio
async def test_prompt_is_untouched_when_there_are_no_tools():
    with _patch_harness():
        await run_agent("what is 2+2?", durable=False, observability=False)

    assert CALLS[0].prompt == "what is 2+2?"


@pytest.mark.asyncio
async def test_prompt_carries_the_tool_manual():
    """The harness has no tool-declaration channel, so the manual rides the prompt."""
    with _patch_harness():
        await run_agent("weather in Paris?", tools=[_weather_tool("ds_run_a")], durable=False, observability=False)

    prompt = CALLS[0].prompt
    assert "get_weather(city: string)" in prompt
    assert f"{TOOLS_DIRNAME}/get_weather" in prompt
    assert prompt.endswith("weather in Paris?")


@pytest.mark.asyncio
async def test_bare_tasks_are_accepted_as_tools():
    env = flyte.TaskEnvironment("ds_run_b")

    @env.task
    async def get_population(city: str) -> int:
        """Get the population of a city."""
        return 1

    with _patch_harness():
        await run_agent("hello", tools=[get_population], durable=False, observability=False)

    assert "get_population(city: string)" in CALLS[0].prompt


@pytest.mark.asyncio
async def test_tools_are_callable_while_the_blocking_run_is_in_flight():
    """The core of the design: ``run`` blocks a worker thread, the loop serves tools."""
    weather = _weather_tool("ds_run_c")

    def call_the_shim(harness):
        shim = pathlib.Path(harness.config.cwd) / TOOLS_DIRNAME / "get_weather"
        return subprocess.run(
            [str(shim), json.dumps({"city": "Paris"})], capture_output=True, text=True, timeout=30, check=False
        )

    FakeHarness.during_run = call_the_shim
    with (
        _patch_harness(),
        patch.object(weather.task, "aio", new_callable=AsyncMock, return_value="sunny in Paris") as mock_aio,
    ):
        await run_agent("weather?", tools=[weather], durable=False, observability=False)

    done = CALLS[0].during_run_result
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "sunny in Paris"
    mock_aio.assert_awaited_once_with(city="Paris")


@pytest.mark.asyncio
async def test_durable_session_id_drives_the_run(tmp_path):
    session = SimpleNamespace(session_id="flyte-abc", resumed=False, persist=AsyncMock())
    with (
        _patch_harness(),
        patch("flyteplugins.agents.deepseek._run.wire_durable_session", AsyncMock(return_value=session)),
    ):
        await run_agent("hello", observability=False)

    assert CALLS[0].session_id == "flyte-abc"
    session.persist.assert_awaited_once()


@pytest.mark.asyncio
async def test_memory_takes_precedence_over_the_per_run_checkpoint():
    """A keyed store survives retries too, so it subsumes crash-resume."""
    memory = SimpleNamespace(session_id="flyte-mem-1", resumed=True, persist=AsyncMock())
    wire_durable = AsyncMock()
    with (
        _patch_harness(),
        patch("flyteplugins.agents.deepseek._run.wire_memory_session", AsyncMock(return_value=memory)),
        patch("flyteplugins.agents.deepseek._run.wire_durable_session", wire_durable),
    ):
        await run_agent("hello", memory_key="user-1", observability=False)

    assert CALLS[0].session_id == "flyte-mem-1"
    wire_durable.assert_not_awaited()
    memory.persist.assert_awaited_once()


@pytest.mark.asyncio
async def test_an_explicit_session_id_wins():
    with _patch_harness():
        await run_agent("hello", session_id="mine", durable=False, observability=False)

    assert CALLS[0].session_id == "mine"


@pytest.mark.asyncio
async def test_the_temp_workspace_is_removed_after_the_run():
    with _patch_harness():
        await run_agent("hello", tools=[_weather_tool("ds_run_d")], durable=False, observability=False)

    assert not pathlib.Path(CALLS[0].config.cwd).exists()
    assert not pathlib.Path(CALLS[0].config.session_root).exists()


@pytest.mark.asyncio
async def test_a_caller_workspace_is_kept_minus_the_shims(tmp_path):
    """Bring your own workspace (e.g. a downloaded Dir): we leave it as we found it."""
    (tmp_path / "repo.txt").write_text("hello")
    with _patch_harness():
        await run_agent(
            "hello", tools=[_weather_tool("ds_run_e")], workspace=tmp_path, durable=False, observability=False
        )

    assert (tmp_path / "repo.txt").read_text() == "hello"
    assert not (tmp_path / TOOLS_DIRNAME).exists()


@pytest.mark.asyncio
async def test_session_events_are_rendered_to_the_report():
    FakeHarness.stream = (
        Notification(
            method="session.event",
            payload={
                "sessionId": "s",
                "event": {
                    "type": "assistant/message",
                    "data": {"message": {"content": [{"type": "text", "text": "thinking out loud"}]}},
                },
            },
        ),
        Notification(
            method="session.event",
            payload={"sessionId": "s", "event": {"type": "turn/end", "data": {"reason": {"kind": "completed"}}}},
        ),
        Notification(method="session.status", payload={"sessionId": "s", "status": "idle"}),
    )
    rows = []

    class RecordingTimeline:
        def heading(self, _text):
            pass

        def row(self, **kwargs):
            rows.append(kwargs)

    with _patch_harness(), patch("flyteplugins.agents.deepseek._run.ReportTimeline", RecordingTimeline):
        await run_agent("hello", durable=False, observability=True)
        await asyncio.sleep(0.05)  # let the marshalled render callbacks land

    labels = [r["label"] for r in rows]
    assert "assistant" in labels
    assert "turn" in labels
    assert labels[-1] == "result"
    assert "thinking out loud" in next(r["detail"] for r in rows if r["label"] == "assistant")


def test_harness_tool_events_are_rendered():
    """The runtime's own tools (bash, editor) surface as tool/call + tool/result."""
    from flyteplugins.agents.deepseek._run import _render_event

    rows = []

    class RecordingTimeline:
        def row(self, **kwargs):
            rows.append(kwargs)

    timeline = RecordingTimeline()
    _render_event(
        timeline,
        {"type": "tool/call", "data": {"name": "bash", "arguments": '{"command": "ls"}', "callId": "c1"}},
    )
    _render_event(timeline, {"type": "tool/result", "data": {"message": "stats.py test_stats.py"}})
    _render_event(
        timeline,
        {"type": "tool/result", "data": {"error": {"name": "bash", "code": "ENOENT"}, "message": "not found"}},
    )

    assert rows[0]["label"] == "bash"
    assert "ls" in rows[0]["detail"]
    assert rows[1]["meta"] == "harness tool result"
    assert rows[1]["error"] is None
    assert rows[2]["error"] == "error"


def test_assistant_usage_is_summarized():
    from flyteplugins.agents.deepseek._run import _render_event

    rows = []

    class RecordingTimeline:
        def row(self, **kwargs):
            rows.append(kwargs)

    _render_event(
        RecordingTimeline(),
        {
            "type": "assistant/message",
            "data": {
                "message": {"content": [{"type": "text", "text": "hi"}]},
                "usage": {"inputTokens": 5000, "outputTokens": 120},
            },
        },
    )
    assert rows[0]["meta"] == "in 5.0k · out 120"


def test_a_malformed_event_never_breaks_the_run():
    """Observability is best-effort: junk on the wire must not raise."""
    from flyteplugins.agents.deepseek._run import _render_event

    class BrokenTimeline:
        def row(self, **kwargs):
            raise RuntimeError("report backend down")

    _render_event(BrokenTimeline(), {"type": "assistant/message", "data": {"message": {"content": [{"text": "x"}]}}})
    _render_event(BrokenTimeline(), {"type": "turn/end", "data": None})
    _render_event(BrokenTimeline(), {})


def test_run_agent_sync_drives_the_same_path():
    """Sync tasks call the sync companion; it runs the async body on its own loop."""
    with _patch_harness():
        assert run_agent_sync("hello", durable=False, observability=False) == "the answer"
