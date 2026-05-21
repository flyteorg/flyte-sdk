"""Tests for the imperative prewarm() method on reusable TaskEnvironments.

Design context: SE-760. The prewarm task is auto-synthesized on every reusable
env; env.prewarm() submits it fire-and-forget so the backend's
`GetOrCreateEnvironment` spins up the actor pool ahead of the first heavy task.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import flyte
from flyte._internal.resolvers.prewarm import (
    PrewarmTaskResolver,
    prewarm_task_full_name,
    prewarm_task_short_name,
)
from flyte._internal.runtime.task_serde import translate_task_to_wire
from flyte.models import SerializationContext


def test_reusable_env_synthesizes_prewarm_task():
    env = flyte.TaskEnvironment(
        name="warm_env",
        reusable=flyte.ReusePolicy(replicas=2, idle_ttl=60),
    )
    expected = prewarm_task_full_name("warm_env")
    assert expected == "warm_env.prewarm_warm_env"  # documents the naming convention
    assert expected in env.tasks
    t = env.tasks[expected]
    assert t.parent_env_name == "warm_env"
    assert t.short_name == prewarm_task_short_name("warm_env")
    assert t.short_name == "prewarm_warm_env"
    assert t.reusable == env.reusable
    assert isinstance(t.task_resolver, PrewarmTaskResolver)


def test_non_reusable_env_skips_prewarm_synthesis():
    env = flyte.TaskEnvironment(name="cold_env")
    assert env.tasks == {}


def test_prewarm_task_shares_actor_version_with_real_task():
    """The prewarm task must hash to the same actor `version` as the env's real
    tasks, otherwise the backend treats them as different pools and the warm-up
    accomplishes nothing."""
    env = flyte.TaskEnvironment(
        name="shared_pool_env",
        reusable=flyte.ReusePolicy(replicas=(1, 3), idle_ttl=120),
    )

    @env.task
    async def heavy(x: int) -> int:
        return x

    # root_dir must be an ancestor of the test file (DefaultTaskResolver requirement).
    repo_root = Path(__file__).resolve().parents[2]
    sctx = SerializationContext(
        project="p",
        domain="d",
        org="o",
        root_dir=repo_root,
        code_bundle=None,
        version="v1",
    )
    heavy_wire = translate_task_to_wire(env.tasks["shared_pool_env.heavy"], sctx)
    prewarm_wire = translate_task_to_wire(env.tasks[prewarm_task_full_name("shared_pool_env")], sctx)
    assert heavy_wire.task_template.custom["version"] == prewarm_wire.task_template.custom["version"], (
        "prewarm version diverged — prewarm would target a different pool"
    )
    assert prewarm_wire.task_template.custom["type"] == "actor"
    assert prewarm_wire.task_template.custom["name"] == "shared_pool_env"


def test_prewarm_resolver_roundtrip():
    env = flyte.TaskEnvironment(
        name="rt_env",
        reusable=flyte.ReusePolicy(replicas=1, idle_ttl=60),
    )
    task = env.tasks[prewarm_task_full_name("rt_env")]
    resolver = PrewarmTaskResolver()
    args = resolver.loader_args(task)
    assert args == ["env_name", "rt_env"]
    reloaded = resolver.load_task(args)
    assert reloaded.name == prewarm_task_full_name("rt_env")
    assert reloaded.short_name == "prewarm_rt_env"
    assert reloaded.parent_env_name == "rt_env"
    assert reloaded.interface.outputs == {"o0": int}
    assert reloaded.interface.inputs == {}


def test_prewarm_resolver_rejects_malformed_args():
    resolver = PrewarmTaskResolver()
    with pytest.raises(ValueError):
        resolver.load_task(["wrong"])
    with pytest.raises(ValueError):
        resolver.load_task(["foo", "bar"])


class _ListHandler(logging.Handler):
    """Capture log records from the propagate=False flyte logger."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def flyte_log_capture():
    flyte_logger = logging.getLogger("flyte")
    handler = _ListHandler()
    flyte_logger.addHandler(handler)
    try:
        yield handler
    finally:
        flyte_logger.removeHandler(handler)


@pytest.mark.asyncio
async def test_prewarm_on_non_reusable_env_warns_and_returns(flyte_log_capture):
    env = flyte.TaskEnvironment(name="cold_env_warn")
    await env.prewarm()
    msgs = [r.getMessage() for r in flyte_log_capture.records]
    assert any("not reusable" in m for m in msgs), msgs


@pytest.mark.asyncio
async def test_prewarm_outside_task_context_warns_and_returns(flyte_log_capture):
    """Calling prewarm() from a bare async function (no TaskContext) is a no-op
    plus warning — there is no controller to submit through."""
    env = flyte.TaskEnvironment(
        name="ctxless_env",
        reusable=flyte.ReusePolicy(replicas=1, idle_ttl=60),
    )
    await env.prewarm()
    msgs = [r.getMessage() for r in flyte_log_capture.records]
    assert any("outside a task context" in m for m in msgs), msgs


def test_prewarm_sync_on_non_reusable_env_warns_and_returns(flyte_log_capture):
    """Sync companion: same warn-and-noop behavior as the async variant."""
    env = flyte.TaskEnvironment(name="cold_env_sync_warn")
    env.prewarm_sync()
    msgs = [r.getMessage() for r in flyte_log_capture.records]
    assert any("not reusable" in m for m in msgs), msgs


def test_prewarm_sync_outside_task_context_warns_and_returns(flyte_log_capture):
    """Sync companion outside a TaskContext also warn-and-noops without raising."""
    env = flyte.TaskEnvironment(
        name="ctxless_sync_env",
        reusable=flyte.ReusePolicy(replicas=1, idle_ttl=60),
    )
    env.prewarm_sync()
    msgs = [r.getMessage() for r in flyte_log_capture.records]
    assert any("outside a task context" in m for m in msgs), msgs
