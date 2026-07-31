"""Unit tests for the durable CrewAI ``LLM`` wrapper (no network)."""

import pytest

import flyteplugins.agents.crewai._durable as durable_mod


def test_fingerprint_is_deterministic_and_tool_order_insensitive():
    tools_a = [{"alpha": object()}, {"beta": object()}]
    tools_b = [{"beta": object()}, {"alpha": object()}]
    msgs = [{"role": "user", "content": "hi"}]
    assert durable_mod._messages_fingerprint(msgs, tools_a) == durable_mod._messages_fingerprint(msgs, tools_b)


def test_fingerprint_changes_with_messages():
    a = durable_mod._messages_fingerprint([{"role": "user", "content": "one"}], None)
    b = durable_mod._messages_fingerprint([{"role": "user", "content": "two"}], None)
    assert a != b


def test_dumps_loads_string_roundtrip():
    assert durable_mod._dumps("plain completion") == "plain completion"
    assert durable_mod._loads("plain completion") == "plain completion"


def test_dumps_loads_structured_roundtrip():
    obj = {"answer": 42, "sources": ["a", "b"]}
    recorded = durable_mod._dumps(obj)
    assert recorded.startswith(durable_mod._JSON_PREFIX)
    assert durable_mod._loads(recorded) == obj


@pytest.fixture
def durable_cls():
    """The durable LLM class built over the concrete provider for gpt-4o."""
    return durable_mod._make_durable_llm_class("gpt-4o")


@pytest.mark.asyncio
async def test_call_routes_through_durable_step_and_replays(durable_cls, monkeypatch):
    """The sync ``call`` (used by kickoff) records its completion via durable_step;
    a second identical call with a populated memo replays without re-calling."""
    provider_cls = durable_cls.__mro__[1]  # the concrete provider class

    inner_calls = {"n": 0}

    def fake_inner_call(self, messages, *a, **k):
        inner_calls["n"] += 1
        return f"COMPLETION-{inner_calls['n']}"

    monkeypatch.setattr(provider_cls, "call", fake_inner_call)

    # Fake durable_step: a memo keyed by request_key, so a repeat replays.
    memo = {}

    async def fake_durable_step(request_key, run, *, name, dumps, loads):
        if request_key in memo:
            return loads(memo[request_key])
        recorded = dumps(await run())
        memo[request_key] = recorded
        return loads(recorded)

    monkeypatch.setattr(durable_mod, "durable_step", fake_durable_step)

    llm = durable_cls(model="gpt-4o")
    msgs = [{"role": "user", "content": "hi"}]

    r1 = llm.call(msgs, None)
    assert r1 == "COMPLETION-1"
    assert inner_calls["n"] == 1

    # Same request -> replays from the memo, inner NOT called again.
    r2 = llm.call(msgs, None)
    assert r2 == "COMPLETION-1"
    assert inner_calls["n"] == 1


@pytest.mark.asyncio
async def test_acall_routes_through_durable_step_and_replays(durable_cls, monkeypatch):
    """The async ``acall`` also records/replays through durable_step."""
    provider_cls = durable_cls.__mro__[1]

    inner_calls = {"n": 0}

    async def fake_inner_acall(self, messages, *a, **k):
        inner_calls["n"] += 1
        return f"ACOMPLETION-{inner_calls['n']}"

    monkeypatch.setattr(provider_cls, "acall", fake_inner_acall)

    memo = {}

    async def fake_durable_step(request_key, run, *, name, dumps, loads):
        if request_key in memo:
            return loads(memo[request_key])
        recorded = dumps(await run())
        memo[request_key] = recorded
        return loads(recorded)

    monkeypatch.setattr(durable_mod, "durable_step", fake_durable_step)

    llm = durable_cls(model="gpt-4o")
    msgs = [{"role": "user", "content": "hi"}]

    r1 = await llm.acall(msgs, None)
    assert r1 == "ACOMPLETION-1"
    assert inner_calls["n"] == 1

    r2 = await llm.acall(msgs, None)
    assert r2 == "ACOMPLETION-1"
    assert inner_calls["n"] == 1


@pytest.mark.asyncio
async def test_call_falls_back_when_durable_step_raises(durable_cls, monkeypatch):
    """Durability never breaks a run: if durable_step blows up, the real call runs."""
    provider_cls = durable_cls.__mro__[1]

    def fake_inner_call(self, messages, *a, **k):
        return "REAL"

    monkeypatch.setattr(provider_cls, "call", fake_inner_call)

    def boom(*a, **k):
        raise RuntimeError("trace layer down")

    monkeypatch.setattr(durable_mod, "durable_step", boom)

    llm = durable_cls(model="gpt-4o")
    assert llm.call([{"role": "user", "content": "hi"}], None) == "REAL"


def test_make_durable_llm_returns_llm_instance():
    """The durable instance is a real CrewAI LLM (a concrete provider subclass)."""
    llm = durable_mod.make_durable_llm("gpt-4o")
    from crewai.llms.base_llm import BaseLLM

    # ``crewai.LLM`` is a factory returning a provider subclass of ``BaseLLM``;
    # our durable class subclasses that concrete provider, so ``BaseLLM`` holds.
    assert isinstance(llm, BaseLLM)
    assert hasattr(llm, "call") and hasattr(llm, "acall")
