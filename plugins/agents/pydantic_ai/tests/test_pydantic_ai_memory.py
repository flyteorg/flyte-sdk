"""Tests for Pydantic AI cross-run memory — session persistence (no network)."""

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

import flyteplugins.agents.pydantic_ai._memory as memory_mod
import flyteplugins.agents.pydantic_ai._run as run_mod


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
        self.saved = 0
        self.read_json = _Aio(lambda path, default=None: self.data.get(path, default))
        self.write_json = _Aio(lambda path, obj, **kw: self.data.__setitem__(path, obj))
        self.save = _Aio(self._save)
        self.extend = self.messages.extend

    def _save(self):
        self.saved += 1


@pytest.mark.asyncio
async def test_resolve_memory_returns_none_without_key():
    store = await memory_mod.resolve_memory(None)
    assert store is None


@pytest.mark.asyncio
async def test_resolve_memory_returns_none_for_empty_key():
    store = await memory_mod.resolve_memory("")
    assert store is None


@pytest.mark.asyncio
async def test_load_history_returns_empty_for_absent_slot():
    store = _FakeStore()
    assert await memory_mod.load_history(store) == []


@pytest.mark.asyncio
async def test_history_round_trips_through_store():
    """Seed the store's history slot, load it, then save an extended history back."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    store = _FakeStore()

    # Seed prior history (as it would have been persisted by a previous run).
    prior = [ModelRequest(parts=[UserPromptPart(content="what's the weather?")])]
    store.data[memory_mod._MEMORY_HISTORY_PATH] = ModelMessagesTypeAdapter.dump_python(prior)

    loaded = await memory_mod.load_history(store)
    assert len(loaded) == 1
    assert isinstance(loaded[0], ModelRequest)

    # A run result exposes ``all_messages()`` — the full (prior + new) history.
    # An assistant turn is a ``ModelResponse`` (``TextPart``), not a ``ModelRequest``.
    new_history = [*prior, ModelResponse(parts=[TextPart(content="sunny")])]

    class _Result:
        def all_messages(self):
            return new_history

    await memory_mod.save_history(store, _Result())

    # History was written back to the same slot and the store was saved.
    written = store.data[memory_mod._MEMORY_HISTORY_PATH]
    assert ModelMessagesTypeAdapter.validate_python(written) == new_history
    assert store.saved == 1


@pytest.mark.asyncio
async def test_run_agent_seeds_message_history_and_saves(monkeypatch):
    """run_agent loads prior history into ``message_history=`` and persists the result."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    store = _FakeStore()
    prior = [ModelRequest(parts=[UserPromptPart(content="prior turn")])]
    store.data[memory_mod._MEMORY_HISTORY_PATH] = ModelMessagesTypeAdapter.dump_python(prior)

    final_history = [*prior, ModelResponse(parts=[TextPart(content="new answer")])]

    class _FakeResult:
        output = "new answer"

        def all_messages(self):
            return final_history

    seen = {}

    class _FakeAgent:
        async def run(self, message, **kwargs):
            seen["message_history"] = kwargs.get("message_history")
            return _FakeResult()

    async def fake_resolve(key):
        return store

    monkeypatch.setattr(run_mod, "resolve_memory", fake_resolve)

    result = await run_mod.run_agent(
        "hi", agent=_FakeAgent(), memory_key="user-123", durable=False, observability=False
    )
    assert result == "new answer"
    # Prior history was seeded into the run.
    assert seen["message_history"] == prior
    # Full history was persisted back.
    written = store.data[memory_mod._MEMORY_HISTORY_PATH]
    assert ModelMessagesTypeAdapter.validate_python(written) == final_history
