"""Tests for CrewAI cross-run memory — transcript persistence (no network)."""

import pytest

import flyteplugins.agents.crewai._memory as memory_mod
import flyteplugins.agents.crewai._run as run_mod


@pytest.mark.asyncio
async def test_resolve_memory_returns_none_without_key():
    store = await memory_mod.resolve_memory(None)
    assert store is None


@pytest.mark.asyncio
async def test_resolve_memory_returns_none_for_empty_key():
    store = await memory_mod.resolve_memory("")
    assert store is None


class _FakeSave:
    def __init__(self, store):
        self._store = store

    async def aio(self):
        self._store.saves += 1


class _FakeJsonSlot:
    """Path-addressed JSON slots backed by an in-memory dict, with syncified io."""

    def __init__(self, store, kind):
        self._store = store
        self._kind = kind

    async def aio(self, path, obj=None, default=None):
        if self._kind == "read":
            return self._store.slots.get(path, default)
        self._store.slots[path] = obj
        return None


class _FakeStore:
    """Minimal stand-in for ``MemoryStore``: path-addressed JSON + a ``save``."""

    def __init__(self):
        self.slots = {}
        self.saves = 0
        self.save = _FakeSave(self)
        self.read_json = _FakeJsonSlot(self, "read")
        self.write_json = _FakeJsonSlot(self, "write")


class _FakeOutput:
    def __init__(self, raw):
        self.raw = raw


class _FakeAgent:
    """A stub agent whose kickoff_async returns a canned ``.raw`` and records input."""

    def __init__(self, raw):
        self._raw = raw
        self.kicked_with = None

    async def kickoff_async(self, messages, **kwargs):
        self.kicked_with = messages
        return _FakeOutput(self._raw)


@pytest.mark.asyncio
async def test_load_build_save_roundtrip():
    """A seeded transcript loads, prepends to the new turn, and both turns persist."""
    store = _FakeStore()
    # Seed a prior transcript.
    store.slots[memory_mod._HISTORY_PATH] = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]

    history = await memory_mod.load_history(store)
    kickoff_input = memory_mod.build_input(history, "second question")
    assert kickoff_input == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]

    await memory_mod.save_turn(store, "second question", "second answer")

    assert store.slots[memory_mod._HISTORY_PATH] == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]
    assert store.saves == 1


@pytest.mark.asyncio
async def test_run_agent_resumes_and_appends_transcript(monkeypatch):
    """run_agent loads prior transcript, kicks off with prior+new, and writes back."""
    store = _FakeStore()
    store.slots[memory_mod._HISTORY_PATH] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    async def _fake_resolve(_key):
        return store

    monkeypatch.setattr(run_mod._memory, "resolve_memory", _fake_resolve)

    agent = _FakeAgent("the answer")
    result = await run_mod.run_agent("new question", agent=agent, memory_key="user-1")

    assert result == "the answer"
    # kickoff saw the prior transcript prepended to the new user turn.
    assert agent.kicked_with == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "new question"},
    ]
    # The new user + assistant turns were appended and persisted.
    assert store.slots[memory_mod._HISTORY_PATH][-2:] == [
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "the answer"},
    ]
    assert store.saves == 1


@pytest.mark.asyncio
async def test_run_agent_without_memory_key_uses_plain_string():
    """No memory_key -> kickoff gets the plain input string, no store touched."""
    agent = _FakeAgent("ok")
    result = await run_mod.run_agent("plain prompt", agent=agent)
    assert result == "ok"
    assert agent.kicked_with == "plain prompt"


@pytest.mark.asyncio
async def test_save_turn_is_best_effort(monkeypatch):
    """A failing write is swallowed (memory never breaks a run)."""

    class _BoomStore(_FakeStore):
        pass

    store = _BoomStore()

    async def _boom(*a, **k):
        raise RuntimeError("network down")

    store.write_json.aio = _boom
    # Should not raise.
    await memory_mod.save_turn(store, "q", "a")
