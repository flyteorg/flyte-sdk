"""Unit tests for flyteplugins-agents-core."""

import json

import flyte
import pytest

from flyteplugins.agents.core import (
    ToolTaskResolver,
    attach_tool_resolver,
    durable_step,
    fingerprint,
)


def test_fingerprint_is_deterministic_and_order_insensitive():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


def test_fingerprint_changes_with_payload():
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


@pytest.mark.asyncio
async def test_durable_step_runs_once_and_round_trips_outside_task_context():
    """Outside a task context flyte.trace is transparent: run once, serde round-trips."""
    calls = {"n": 0}

    async def run():
        calls["n"] += 1
        return {"value": 42}

    out = await durable_step("key-1", run, dumps=json.dumps, loads=json.loads)

    assert calls["n"] == 1
    assert out == {"value": 42}


@pytest.mark.asyncio
async def test_durable_step_default_serde_is_identity():
    async def run():
        return "already-a-string"

    out = await durable_step("key-2", run)
    assert out == "already-a-string"


def test_attach_tool_resolver_wires_resolver():
    env = flyte.TaskEnvironment("core-resolver")

    @env.task
    def my_task(x: int) -> int:
        """A task."""
        return x

    attach_tool_resolver(my_task)
    assert isinstance(my_task.task_resolver, ToolTaskResolver)


def test_attach_tool_resolver_is_noop_for_non_tasks():
    # A plain object must not raise and must not gain a resolver.
    attach_tool_resolver(object())
