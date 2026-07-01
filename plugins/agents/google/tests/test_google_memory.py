"""Tests for Google ADK cross-run memory — session-event persistence (no network)."""

import pytest

import flyteplugins.agents.google._memory as memory_mod
from flyteplugins.agents.google._memory import load_memory, save_memory


class _Aio:
    def __init__(self, fn):
        self._fn = fn

    async def aio(self, *a, **k):
        return self._fn(*a, **k)


class _FakeStore:
    """In-memory ``MemoryStore`` stand-in (path-addressed JSON)."""

    def __init__(self):
        self.data: dict = {}
        self.read_json = _Aio(lambda path, default=None: self.data.get(path, default))
        self.write_json = _Aio(lambda path, obj, **kw: self.data.__setitem__(path, obj))
        self.save = _Aio(lambda: None)


@pytest.mark.asyncio
async def test_events_round_trip_across_runs(monkeypatch):
    from google.adk.events import Event
    from google.genai import types as gt

    store = _FakeStore()
    events = [
        Event(author="user", content=gt.Content(role="user", parts=[gt.Part.from_text(text="my name is Alice")])),
        Event(author="agent", content=gt.Content(role="model", parts=[gt.Part.from_text(text="Hi Alice!")])),
    ]
    await save_memory(store, events)

    # A later run resolves the same store and restores the prior events.
    async def fake_resolve(key, **kw):
        return store

    monkeypatch.setattr(memory_mod, "resolve_memory", fake_resolve)
    loaded_store, loaded = await load_memory("alice")

    assert loaded_store is store
    assert [e.author for e in loaded] == ["user", "agent"]
    assert loaded[0].content.parts[0].text == "my name is Alice"


@pytest.mark.asyncio
async def test_load_memory_is_noop_without_key():
    # resolve_memory(None) returns None, so memory degrades to empty without raising.
    store, events = await load_memory(None)
    assert store is None
    assert events == []


@pytest.mark.asyncio
async def test_save_memory_noop_when_store_is_none():
    await save_memory(None, [])  # must not raise
