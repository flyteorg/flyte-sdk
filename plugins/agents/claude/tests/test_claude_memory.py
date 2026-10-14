"""Tests for Claude cross-run memory — a MemoryStore-backed SessionStore.

Offline: a ``_FakeStore`` stands in for the keyed ``MemoryStore`` (path-addressed
``read_json``/``write_json``/``save``), so we exercise the transcript round-trip and
the first-run-pins / later-run-resumes wiring without a backend.
"""

import pytest
from claude_agent_sdk import ClaudeAgentOptions

import flyteplugins.agents.claude._memory as memory_mod
from flyteplugins.agents.claude._memory import MemorySessionStore, memory_session_id, wire_memory_session

KEY = {"project_key": "p", "session_id": "s1"}


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


def test_memory_session_id_is_stable_and_per_key():
    assert memory_session_id("alice") == memory_session_id("alice")  # stable across runs
    assert memory_session_id("alice") != memory_session_id("bob")


@pytest.mark.asyncio
async def test_transcript_round_trips_across_runs_with_dedup():
    store = _FakeStore()
    first = MemorySessionStore(store)
    assert await first.seed() is False
    await first.append(KEY, [{"type": "user", "uuid": "u1"}])
    await first.append(KEY, [{"type": "user", "uuid": "u1"}])  # duplicate -> ignored
    await first.append(KEY, [{"type": "assistant", "uuid": "u2"}])

    # a fresh session over the same store seeds the persisted transcript
    second = MemorySessionStore(store)
    assert await second.seed() is True
    assert await second.load(KEY) == [{"type": "user", "uuid": "u1"}, {"type": "assistant", "uuid": "u2"}]


@pytest.mark.asyncio
async def test_wire_first_run_pins_session_id_then_resumes(monkeypatch):
    store = _FakeStore()

    async def fake_resolve(key, **kw):
        return store

    monkeypatch.setattr(memory_mod, "resolve_memory", fake_resolve)

    # first run for the key: no prior -> pin a deterministic session_id
    opts1 = ClaudeAgentOptions()
    session = await wire_memory_session(opts1, memory_key="alice")
    assert opts1.session_id == memory_session_id("alice")
    assert opts1.resume is None
    assert opts1.session_store is session
    await session.append(KEY, [{"type": "user", "uuid": "u1"}])

    # later run, same key: prior transcript exists -> resume it
    opts2 = ClaudeAgentOptions()
    await wire_memory_session(opts2, memory_key="alice")
    assert opts2.resume == memory_session_id("alice")
    assert opts2.session_id is None


@pytest.mark.asyncio
async def test_wire_is_noop_without_memory_key():
    opts = ClaudeAgentOptions()
    assert await wire_memory_session(opts, memory_key=None) is None
    assert opts.session_store is None


@pytest.mark.asyncio
async def test_list_subkeys_returns_subagent_subpaths_scoped_to_the_session():
    # Mirrors the SDK's session-store conformance contract for list_subkeys: it must
    # return the session's subagent subpaths and not leak across sessions.
    store = MemorySessionStore(_FakeStore())
    main = {"project_key": "p", "session_id": "s1"}
    await store.append(main, [{"type": "user", "uuid": "u0"}])
    await store.append({**main, "subpath": "subagents/agent-1"}, [{"type": "x", "uuid": "u1"}])
    await store.append({**main, "subpath": "subagents/agent-2"}, [{"type": "x", "uuid": "u2"}])
    # a subagent under a DIFFERENT session must not leak in
    await store.append({"project_key": "p", "session_id": "s2", "subpath": "subagents/agent-x"}, [{"uuid": "u3"}])

    subkeys = await store.list_subkeys(main)
    assert sorted(subkeys) == ["subagents/agent-1", "subagents/agent-2"]
    assert "subagents/agent-x" not in subkeys


@pytest.mark.asyncio
async def test_list_subkeys_excludes_the_main_transcript():
    store = MemorySessionStore(_FakeStore())
    main = {"project_key": "p", "session_id": "s1"}
    await store.append(main, [{"type": "user", "uuid": "u0"}])
    assert await store.list_subkeys(main) == []
