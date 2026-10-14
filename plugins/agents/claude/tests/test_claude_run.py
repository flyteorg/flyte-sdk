"""Sync-form smoke test for ``run_agent_sync`` (no CLI subprocess).

``run_agent`` is async: async callers use ``await run_agent(...)``, sync callers
use ``run_agent_sync(...)``. This drives the sync variant end to end with the
SDK's ``query`` stream stubbed out, so no Claude Code subprocess is spawned.
"""

import inspect

import flyteplugins.agents.claude._run as run_mod


class _FakeResultMessage:
    def __init__(self, result):
        self.result = result


def test_run_agent_sync_variant():
    assert inspect.iscoroutinefunction(run_mod.run_agent)
    assert callable(run_mod.run_agent_sync)


def test_run_agent_sync_call(monkeypatch):
    """The sync variant drives the query stream and returns the final text."""

    async def fake_query(*, prompt, options):
        yield _FakeResultMessage("Hello from the sync form.")

    monkeypatch.setattr(run_mod, "query", fake_query)
    monkeypatch.setattr(run_mod, "ResultMessage", _FakeResultMessage)

    out = run_mod.run_agent_sync("say hi", durable=False, observability=False)
    assert out == "Hello from the sync form."


def test_a_registered_call_wrapper_reaches_the_sync_variant(monkeypatch):
    """run_agent_sync dispatches onto a background event loop on another thread.

    The wrapper is resolved inside the coroutine, so it has to survive that hop — and it reads
    the Flyte run from a contextvar, which is exactly the kind of thing a thread boundary
    tends to drop.
    """
    from flyteplugins.agents.core import register_call_wrapper, unregister_call_wrapper

    seen = {}

    async def fake_query(*, prompt, options):
        yield _FakeResultMessage("wrapped result")

    def wrapper(call):
        seen["wrapped"] = call

        async def instrumented(*, prompt, options):
            seen["prompt"] = prompt
            async for message in call(prompt=prompt, options=options):
                yield message

        return instrumented

    monkeypatch.setattr(run_mod, "query", fake_query)
    monkeypatch.setattr(run_mod, "ResultMessage", _FakeResultMessage)
    register_call_wrapper("claude", wrapper)
    try:
        out = run_mod.run_agent_sync("say hi", durable=False, observability=False)
    finally:
        unregister_call_wrapper("claude")

    assert out == "wrapped result"
    assert seen["wrapped"] is fake_query, "the SDK's own query must be threaded through"
    assert seen["prompt"] == "say hi"


def test_no_call_wrapper_leaves_the_sync_path_untouched(monkeypatch):
    async def fake_query(*, prompt, options):
        yield _FakeResultMessage("plain result")

    monkeypatch.setattr(run_mod, "query", fake_query)
    monkeypatch.setattr(run_mod, "ResultMessage", _FakeResultMessage)
    assert run_mod.run_agent_sync("hi", durable=False, observability=False) == "plain result"
