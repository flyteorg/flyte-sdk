"""Tests for LangChain cross-run memory — session persistence (no network)."""

import pytest

import flyteplugins.agents.langchain._memory as memory_mod


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
async def test_load_history_empty_store_returns_empty_list():
    store = _FakeStore()
    assert await memory_mod.load_history(store) == []


@pytest.mark.asyncio
async def test_load_history_none_store_returns_empty_list():
    assert await memory_mod.load_history(None) == []


@pytest.mark.asyncio
async def test_history_round_trip():
    """save_history + load_history round-trip LangChain messages through the store."""
    from langchain_core.messages import AIMessage, HumanMessage

    store = _FakeStore()
    messages = [HumanMessage(content="hi"), AIMessage(content="hello there")]

    await memory_mod.save_history(store, messages)
    # The transcript is persisted to the path-addressed history slot.
    assert memory_mod._HISTORY_PATH in store.data

    loaded = await memory_mod.load_history(store)
    assert [m.content for m in loaded] == ["hi", "hello there"]
    assert [m.type for m in loaded] == ["human", "ai"]


@pytest.mark.asyncio
async def test_save_history_ignores_empty_messages():
    """An empty transcript must not clobber a prior conversation in the store."""
    from langchain_core.messages import HumanMessage

    store = _FakeStore()
    await memory_mod.save_history(store, [HumanMessage(content="keep me")])
    await memory_mod.save_history(store, [])  # no-op

    loaded = await memory_mod.load_history(store)
    assert [m.content for m in loaded] == ["keep me"]
