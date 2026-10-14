"""Tests for the core agent-memory helper (best-effort, never fatal)."""

import pytest

from flyteplugins.agents.core import resolve_memory


@pytest.mark.asyncio
async def test_resolve_memory_none_for_empty_key():
    assert await resolve_memory(None) is None
    assert await resolve_memory("") is None


@pytest.mark.asyncio
async def test_resolve_memory_best_effort_without_context():
    # With no Flyte run context / org configured the keyed store can't resolve;
    # we degrade to None rather than raising, so memory never breaks a run.
    assert await resolve_memory("a-key") is None
