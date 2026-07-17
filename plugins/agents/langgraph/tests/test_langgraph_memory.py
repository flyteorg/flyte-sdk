"""Tests for LangGraph cross-run memory — session persistence (no network)."""

import pytest

import flyteplugins.agents.langgraph._memory as memory_mod


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
async def test_load_messages_empty_store():
    assert await memory_mod.load_messages(None) == []
    assert await memory_mod.load_messages(_FakeStore()) == []


@pytest.mark.asyncio
async def test_messages_round_trip():
    from langchain_core.messages import AIMessage, HumanMessage

    store = _FakeStore()
    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    await memory_mod.save_messages(store, messages)

    # The transcript is serialized to the history slot and survives a fresh load.
    assert memory_mod._MEMORY_HISTORY_PATH in store.data
    loaded = await memory_mod.load_messages(store)
    assert [m.content for m in loaded] == ["hi", "hello"]
    assert [type(m).__name__ for m in loaded] == ["HumanMessage", "AIMessage"]
