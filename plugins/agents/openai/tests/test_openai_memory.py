"""Tests for the OpenAI MemoryStore-backed ``Session`` (no store/context needed)."""

import pytest

from flyteplugins.agents.openai import FlyteSession


class _FakeSave:
    def __init__(self, store):
        self._store = store

    async def aio(self):
        self._store.saves += 1


class _FakeStore:
    """Minimal stand-in for ``MemoryStore``: a ``messages`` list + a ``save``."""

    def __init__(self):
        self.messages = []
        self.saves = 0
        self.save = _FakeSave(self)

    def extend(self, items):
        self.messages.extend(items)


@pytest.mark.asyncio
async def test_flyte_session_persists_and_reads_back():
    store = _FakeStore()
    session = FlyteSession(store)

    assert await session.get_items() == []
    await session.add_items([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])

    assert store.messages == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    assert store.saves == 1  # add_items persists durably (an object-store upload)
    assert await session.get_items() == store.messages
    assert await session.get_items(limit=1) == [{"role": "assistant", "content": "yo"}]


@pytest.mark.asyncio
async def test_flyte_session_pop_and_clear():
    store = _FakeStore()
    store.messages = [{"a": 1}, {"b": 2}]
    session = FlyteSession(store)

    assert await session.pop_item() == {"b": 2}
    assert store.messages == [{"a": 1}]
    await session.clear_session()
    assert store.messages == []
    assert await session.pop_item() is None  # empty
