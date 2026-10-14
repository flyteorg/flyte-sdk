"""Threading the caller's RunHooks through to any registered instrumentor."""

from types import SimpleNamespace

import pytest

import flyteplugins.agents.openai._run as run_mod


@pytest.mark.asyncio
async def test_caller_hooks_are_offered_to_the_instrumentor(monkeypatch):
    """The injector chains onto an existing RunHooks; it cannot if handed nothing."""
    from flyteplugins.agents.core import register_instrumentor, unregister_instrumentor

    seen = {}

    class FakeRunner:
        @staticmethod
        async def run(agent, input, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(final_output="done")

    def chain(run_options):
        merged = dict(run_options or {})
        merged["hooks"] = f"WRAPPED({merged.get('hooks')})"
        return merged

    monkeypatch.setattr(run_mod, "Runner", FakeRunner)
    register_instrumentor("openai", chain)
    try:
        await run_mod.run_agent("hi", model="gpt-4.1", observability=False, durable=False, hooks="MINE")
    finally:
        unregister_instrumentor("openai")

    assert seen["hooks"] == "WRAPPED(MINE)"


@pytest.mark.asyncio
async def test_caller_hooks_are_passed_with_no_instrumentor(monkeypatch):
    seen = {}

    class FakeRunner:
        @staticmethod
        async def run(agent, input, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(final_output="done")

    monkeypatch.setattr(run_mod, "Runner", FakeRunner)
    await run_mod.run_agent("hi", model="gpt-4.1", observability=False, durable=False, hooks="MINE")
    assert seen["hooks"] == "MINE"


@pytest.mark.asyncio
async def test_no_hooks_and_no_instrumentor_passes_nothing(monkeypatch):
    """The uninstrumented path must stay exactly as it was."""
    seen = {}

    class FakeRunner:
        @staticmethod
        async def run(agent, input, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(final_output="done")

    monkeypatch.setattr(run_mod, "Runner", FakeRunner)
    await run_mod.run_agent("hi", model="gpt-4.1", observability=False, durable=False)
    assert "hooks" not in seen
