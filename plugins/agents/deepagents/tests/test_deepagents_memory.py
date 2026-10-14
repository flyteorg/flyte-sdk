"""Tests for deepagents cross-run memory — conversation + files persistence (no network)."""

import pytest

import flyteplugins.agents.deepagents._memory as memory_mod


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
async def test_load_empty_store():
    assert await memory_mod.load_history(None) == []
    assert await memory_mod.load_history(_FakeStore()) == []
    assert await memory_mod.load_files(None) == {}
    assert await memory_mod.load_files(_FakeStore()) == {}


@pytest.mark.asyncio
async def test_history_and_files_round_trip():
    from langchain_core.messages import AIMessage, HumanMessage

    store = _FakeStore()
    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    files = {"notes.txt": "remember this"}
    await memory_mod.save_state(store, messages, files=files)

    # Both slots are populated and survive a fresh load.
    assert memory_mod._HISTORY_PATH in store.data
    assert memory_mod._FILES_PATH in store.data
    loaded = await memory_mod.load_history(store)
    assert [m.content for m in loaded] == ["hi", "hello"]
    assert [type(m).__name__ for m in loaded] == ["HumanMessage", "AIMessage"]
    assert await memory_mod.load_files(store) == files


@pytest.mark.asyncio
async def test_save_state_without_files_keeps_prior_files():
    from langchain_core.messages import HumanMessage

    store = _FakeStore()
    await memory_mod.save_state(store, [HumanMessage(content="a")], files={"f.txt": "v1"})
    # A later save with no files must not clobber the stored filesystem.
    await memory_mod.save_state(store, [HumanMessage(content="b")], files=None)
    assert await memory_mod.load_files(store) == {"f.txt": "v1"}
