"""Unit tests for Google ADK run helpers — final-text extraction + timeline rendering."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from google.genai import types as gt

from flyteplugins.agents.google._run import _content_text, _render, _run_config, _UsageSink


def test_content_text_joins_text_parts():
    content = gt.Content(role="model", parts=[gt.Part.from_text(text="Hello "), gt.Part.from_text(text="world")])
    assert _content_text(content) == "Hello world"


def test_content_text_empty_for_no_parts():
    assert _content_text(MagicMock(parts=None)) == ""


def test_render_emits_rows_for_tool_calls_and_assistant_text():
    timeline = MagicMock()

    call = MagicMock(args={"city": "SF"})
    call.name = "get_weather"
    call_part = MagicMock(function_call=call, function_response=None, text=None)
    text_part = MagicMock(function_call=None, function_response=None, text="The weather is sunny")
    event = MagicMock(content=MagicMock(parts=[call_part, text_part]))

    _render(timeline, event)

    labels = [c.kwargs.get("label") for c in timeline.row.call_args_list]
    assert "get_weather" in labels  # tool call rendered
    assert "assistant" in labels  # assistant text rendered


def test_run_config_caps_llm_calls():
    # None -> let ADK use its own default (500); a value -> a real RunConfig cap.
    assert _run_config(None) is None
    rc = _run_config(7)
    assert rc.max_llm_calls == 7


def _usage_meta(prompt, completion, total, cached=0, thoughts=0):
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=completion,
        total_token_count=total,
        cached_content_token_count=cached,
        thoughts_token_count=thoughts,
    )


def test_usage_sink_tallies_tokens_and_skips_non_model_events():
    sink = _UsageSink()
    sink.add(SimpleNamespace(usage_metadata=_usage_meta(100, 20, 120)))  # model turn
    sink.add(SimpleNamespace(usage_metadata=None))  # tool-result event: no usage, not a turn
    sink.add(SimpleNamespace(usage_metadata=_usage_meta(50, 10, 60, cached=40)))  # model turn w/ cache

    assert sink.turns == 2
    assert (sink.prompt, sink.completion, sink.total) == (150, 30, 180)
    detail = sink.detail()
    assert "2 model turns" in detail and "180 total tokens" in detail
    assert "40 cached" in detail  # Gemini context-cache tokens surfaced


def test_usage_sink_omits_cached_and_thinking_when_zero():
    sink = _UsageSink()
    sink.add(SimpleNamespace(usage_metadata=_usage_meta(100, 20, 120)))
    detail = sink.detail()
    assert "cached" not in detail and "thinking" not in detail


def test_default_agent_name_is_natural():
    # ADK injects the agent name into the system prompt as the model's "internal name"
    # (identity.py), so the default must read naturally if the model mentions it — not a
    # brand-y identifier like "flyte_agent".
    import inspect

    from flyteplugins.agents.google._run import run_agent

    assert inspect.signature(run_agent).parameters["name"].default == "assistant"


def test_run_agent_sync_call():
    """run_agent is syncified: the plain sync form drives the loop end to end."""
    from unittest.mock import patch

    from flyteplugins.agents.google import run_agent

    final_event = MagicMock()
    final_event.is_final_response.return_value = True
    final_event.content = gt.Content(role="model", parts=[gt.Part.from_text(text="Hello from the sync form.")])

    async def fake_run_async(**kwargs):
        yield final_event

    fake_runner = MagicMock()
    fake_runner.run_async = fake_run_async

    with patch("google.adk.runners.Runner", return_value=fake_runner):
        out = run_agent("say hi", model="gemini-2.0-flash", durable=False, observability=False)

    assert out == "Hello from the sync form."
