"""Tests for flyte.ai.agents._llm (default litellm-backed LLM callback).

These exercise the provider-response → :class:`LLMMessage` normalization
without hitting a real model: ``litellm.acompletion`` is patched with an
``AsyncMock`` returning hand-built response objects.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock

import pytest

from flyte.ai.agents import LLMMessage
from flyte.ai.agents._llm import _default_call_llm


class _Func:
    def __init__(self, name: str, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, name: str, arguments, *, id=None):
        self.id = id
        self.function = _Func(name, arguments)


class _Message:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message: _Message):
        self.message = message


class _Response:
    def __init__(self, message: _Message):
        self.choices = [_Choice(message)]


def _patch_litellm(monkeypatch: pytest.MonkeyPatch, response: _Response) -> AsyncMock:
    """Install a fake ``litellm`` module exposing an ``acompletion`` AsyncMock."""
    acompletion = AsyncMock(return_value=response)
    fake = types.ModuleType("litellm")
    fake.acompletion = acompletion  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", fake)
    return acompletion


@pytest.mark.asyncio
class TestDefaultCallLLM:
    async def test_plain_text_response(self, monkeypatch: pytest.MonkeyPatch):
        acompletion = _patch_litellm(monkeypatch, _Response(_Message("hello there")))
        out = await _default_call_llm("gpt-x", "system prompt", [{"role": "user", "content": "hi"}], None)
        assert isinstance(out, LLMMessage)
        assert out.content == "hello there"
        assert out.tool_calls == []
        assert out.raw is not None

        # System prompt is prepended; no tools => no tool kwargs.
        kwargs = acompletion.await_args.kwargs
        assert kwargs["messages"][0] == {"role": "system", "content": "system prompt"}
        assert kwargs["messages"][1] == {"role": "user", "content": "hi"}
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs

    async def test_tools_forwarded_with_auto_choice(self, monkeypatch: pytest.MonkeyPatch):
        acompletion = _patch_litellm(monkeypatch, _Response(_Message("ok")))
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        await _default_call_llm("m", "s", [{"role": "user", "content": "x"}], tools)
        kwargs = acompletion.await_args.kwargs
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == "auto"

    async def test_tool_calls_parsed_from_json_arguments(self, monkeypatch: pytest.MonkeyPatch):
        msg = _Message(None, tool_calls=[_ToolCall("search", '{"q": "cats"}', id="call_1")])
        _patch_litellm(monkeypatch, _Response(msg))
        out = await _default_call_llm("m", "s", [], None)
        # content None normalizes to empty string.
        assert out.content == ""
        assert out.tool_calls == [{"id": "call_1", "name": "search", "arguments": {"q": "cats"}}]

    async def test_dict_arguments_passed_through(self, monkeypatch: pytest.MonkeyPatch):
        # Some providers hand back already-decoded dict arguments.
        msg = _Message("", tool_calls=[_ToolCall("f", {"a": 1}, id="c")])
        _patch_litellm(monkeypatch, _Response(msg))
        out = await _default_call_llm("m", "s", [], None)
        assert out.tool_calls[0]["arguments"] == {"a": 1}

    async def test_malformed_json_arguments_fall_back_to_raw(self, monkeypatch: pytest.MonkeyPatch):
        msg = _Message(None, tool_calls=[_ToolCall("f", "{not valid json", id="c")])
        _patch_litellm(monkeypatch, _Response(msg))
        out = await _default_call_llm("m", "s", [], None)
        assert out.tool_calls[0]["arguments"] == {"_raw": "{not valid json"}

    async def test_missing_tool_call_id_is_generated(self, monkeypatch: pytest.MonkeyPatch):
        msg = _Message(None, tool_calls=[_ToolCall("f", "{}", id=None)])
        _patch_litellm(monkeypatch, _Response(msg))
        out = await _default_call_llm("m", "s", [], None)
        generated = out.tool_calls[0]["id"]
        assert generated.startswith("call_")
        assert len(generated) > len("call_")

    async def test_missing_litellm_raises_helpful_error(self, monkeypatch: pytest.MonkeyPatch):
        # Force ``from litellm import acompletion`` to raise ImportError.
        monkeypatch.setitem(sys.modules, "litellm", None)
        with pytest.raises(ImportError, match="litellm is not installed"):
            await _default_call_llm("m", "s", [], None)
