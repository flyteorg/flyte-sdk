"""Tests for cross-run memory — the session archive in a keyed MemoryStore.

Offline: a ``FakeStore`` stands in for ``flyte.ai.agents.memory.MemoryStore``,
exposing the ``read_json`` / ``write_json`` / ``save`` surface the adapter uses
(each with the ``.aio`` companion the real store has).
"""

from unittest.mock import patch

import pytest

from flyteplugins.agents.deepseek._memory import MemorySession, memory_session_id, wire_memory_session


class _Method:
    """A store method exposing the ``.aio`` companion the real MemoryStore has."""

    def __init__(self, fn):
        self.aio = fn


class FakeStore:
    def __init__(self, initial=None):
        self.data = dict(initial or {})
        self.saved = 0
        self.read_json = _Method(self._read_json)
        self.write_json = _Method(self._write_json)
        self.save = _Method(self._save)

    async def _read_json(self, path, default=None):
        return self.data.get(path, default)

    async def _write_json(self, path, value, actor=None):
        self.data[path] = value
        self.actor = actor

    async def _save(self):
        self.saved += 1


def test_memory_session_id_is_stable_per_key():
    assert memory_session_id("user-1") == memory_session_id("user-1")
    assert memory_session_id("user-1") != memory_session_id("user-2")


@pytest.mark.asyncio
async def test_session_continues_across_runs(tmp_path):
    """Run 1 persists the transcript; run 2 with the same key resumes it."""
    store = FakeStore()

    first_root = tmp_path / "run1"
    first_root.mkdir()
    (first_root / "s.jsonl").write_text('{"turn": 1}\n')
    await MemorySession(store, "sid").persist(first_root)
    assert store.saved == 1

    second_root = tmp_path / "run2"
    second_root.mkdir()
    later = MemorySession(store, "sid")
    assert await later.seed(second_root) is True
    assert (second_root / "s.jsonl").read_text() == '{"turn": 1}\n'


@pytest.mark.asyncio
async def test_first_run_for_a_key_has_nothing_to_seed(tmp_path):
    session = MemorySession(FakeStore(), "sid")
    assert await session.seed(tmp_path) is False
    assert session.resumed is False


@pytest.mark.asyncio
async def test_persist_of_an_empty_session_writes_nothing(tmp_path):
    store = FakeStore()
    await MemorySession(store, "sid").persist(tmp_path)
    assert store.data == {}
    assert store.saved == 0


@pytest.mark.asyncio
async def test_memory_failures_never_break_the_run(tmp_path):
    class BrokenStore(FakeStore):
        async def _read_json(self, path, default=None):
            raise RuntimeError("store unavailable")

        async def _write_json(self, path, value, actor=None):
            raise RuntimeError("store unavailable")

    (tmp_path / "s.jsonl").write_text("{}\n")
    session = MemorySession(BrokenStore(), "sid")
    assert await session.seed(tmp_path) is False
    await session.persist(tmp_path)  # must not raise


@pytest.mark.asyncio
async def test_wire_uses_the_key_derived_session_id(tmp_path):
    store = FakeStore()
    with patch("flyteplugins.agents.deepseek._memory.resolve_memory", return_value=store):
        session = await wire_memory_session(tmp_path, memory_key="user-1")
    assert session is not None
    assert session.session_id == memory_session_id("user-1")


@pytest.mark.asyncio
async def test_wire_is_noop_without_a_key(tmp_path):
    assert await wire_memory_session(tmp_path, memory_key=None) is None


@pytest.mark.asyncio
async def test_wire_is_noop_when_no_store_is_available(tmp_path):
    """No Flyte context/org: the run proceeds without cross-run memory."""
    with patch("flyteplugins.agents.deepseek._memory.resolve_memory", return_value=None):
        assert await wire_memory_session(tmp_path, memory_key="user-1") is None
