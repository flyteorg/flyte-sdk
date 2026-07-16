"""Sync-form smoke test for the syncified ``run_agent`` (no CLI subprocess).

``run_agent`` is syncified: async callers use ``await run_agent.aio(...)``, sync
callers use ``run_agent(...)``. This drives the sync form end to end with the
SDK's ``query`` stream stubbed out, so no Claude Code subprocess is spawned.
"""

import inspect

import flyteplugins.agents.claude._run as run_mod


class _FakeResultMessage:
    def __init__(self, result):
        self.result = result


def test_run_agent_is_syncified():
    assert callable(run_mod.run_agent)
    assert callable(run_mod.run_agent.aio)
    assert inspect.iscoroutinefunction(run_mod.run_agent.fn)


def test_run_agent_sync_call(monkeypatch):
    """The plain sync form drives the query stream and returns the final text."""

    async def fake_query(*, prompt, options):
        yield _FakeResultMessage("Hello from the sync form.")

    monkeypatch.setattr(run_mod, "query", fake_query)
    monkeypatch.setattr(run_mod, "ResultMessage", _FakeResultMessage)

    out = run_mod.run_agent("say hi", durable=False, observability=False)
    assert out == "Hello from the sync form."
