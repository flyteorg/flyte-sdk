"""Tests for Hermes cross-run memory — session persistence (no network)."""

import pytest

import flyteplugins.agents.hermes._memory as memory_mod


class _Aio:
    def __init__(self, fn):
        self._fn = fn

    async def aio(self, *a, **k):
        return self._fn(*a, **k)


class _FakeStore:
    """In-memory ``MemoryStore`` stand-in (path-addressed JSON)."""

    def __init__(self):
        self.data: dict = {}
        self.messages = []
        self.read_json = _Aio(lambda path, default=None: self.data.get(path, default))
        self.write_json = _Aio(lambda path, obj, **kw: self.data.__setitem__(path, obj))
        self.save = _Aio(lambda: None)
        self.extend = self.messages.extend


@pytest.mark.asyncio
async def test_resolve_memory_returns_none_without_key():
    store = await memory_mod.resolve_memory(None)
    assert store is None


@pytest.mark.asyncio
async def test_resolve_memory_returns_none_for_empty_key():
    store = await memory_mod.resolve_memory("")
    assert store is None


@pytest.mark.asyncio
async def test_transcript_round_trip():
    store = _FakeStore()
    assert await memory_mod.load_transcript(None) == []
    assert await memory_mod.load_transcript(store) == []

    transcript = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    await memory_mod.save_transcript(store, transcript)

    assert store.data[memory_mod._MEMORY_HISTORY_PATH] == transcript
    assert await memory_mod.load_transcript(store) == transcript
