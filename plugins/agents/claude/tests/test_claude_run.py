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
