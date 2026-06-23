"""Unit tests for Google ADK run helpers — final-text extraction + timeline rendering."""

from unittest.mock import MagicMock

from google.genai import types as gt

from flyteplugins.agents.google._run import _content_text, _render


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
