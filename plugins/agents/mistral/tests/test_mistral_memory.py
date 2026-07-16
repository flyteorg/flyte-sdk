"""Tests for Mistral cross-run memory — persisted ``conversation_id`` continuation.

Offline: ``resolve_memory``, the Mistral client, and ``RunContext`` are mocked, so we
verify that a stored conversation id is fed into the runner (to continue) and the new
id is persisted afterward — without any network.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import flyteplugins.agents.mistral._run as run_mod
from flyteplugins.agents.mistral import run_agent


class _Aio:
    def __init__(self, fn):
        self._fn = fn

    async def aio(self, *a, **k):
        return self._fn(*a, **k)


def _message(text):
    e = MagicMock()
    e.type = "message.output"
    e.content = text
    return e


@pytest.mark.asyncio
async def test_memory_key_continues_prior_conversation_and_persists_new(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    writes: dict = {}

    class _Store:
        def __init__(self):
            self.read_json = _Aio(lambda path, default=None: "conv-prev")
            self.write_json = _Aio(lambda path, obj, **kw: writes.__setitem__(path, obj))
            self.save = _Aio(lambda: None)

    async def fake_resolve(key, **kw):
        return _Store()

    monkeypatch.setattr(run_mod, "resolve_memory", fake_resolve)

    captured: dict = {}

    class _FakeRunCtx:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def register_func(self, fn):
            pass

    result = MagicMock()
    result.output_entries = [_message("hi Alice")]
    result.conversation_id = "conv-new"
    client = MagicMock()
    client.beta.conversations.run_async = AsyncMock(return_value=result)

    with (
        patch("mistralai.client.Mistral", return_value=client),
        patch("mistralai.extra.run.context.RunContext", _FakeRunCtx),
    ):
        out = await run_agent.aio("hi", durable=False, observability=False, memory_key="t1")

    assert out == "hi Alice"
    assert captured.get("conversation_id") == "conv-prev"  # continued the prior conversation
    assert writes.get("mistral/conversation_id") == "conv-new"  # persisted the new id


@pytest.mark.asyncio
async def test_no_memory_key_starts_fresh(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    captured: dict = {}

    class _FakeRunCtx:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def register_func(self, fn):
            pass

    result = MagicMock()
    result.output_entries = [_message("hello")]
    result.conversation_id = "conv-new"
    client = MagicMock()
    client.beta.conversations.run_async = AsyncMock(return_value=result)

    with (
        patch("mistralai.client.Mistral", return_value=client),
        patch("mistralai.extra.run.context.RunContext", _FakeRunCtx),
    ):
        out = await run_agent.aio("hi", durable=False, observability=False)

    assert out == "hello"
    assert "conversation_id" not in captured  # fresh conversation, no continuation
