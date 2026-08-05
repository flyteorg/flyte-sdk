"""Unit tests for _Runner._resolve_spawn_parent: the implicit spawn-provenance parent
(Relation.related_to) resolution rules. These run on any flyteidl2 build — the resolver only
constructs a RunIdentifier; the descriptor-gated stamping (relation_type=SPAWN) is covered in
test_run_runspec_chars.py."""

import pytest
from flyteidl2.common import identifier_pb2

from flyte._context import Context, ContextData
from flyte._initialize import _init_for_testing
from flyte._run import _Runner
from flyte.models import ActionID, RawDataPath, TaskContext
from flyte.report import Report


def _fake_task_ctx(mode="remote", org="o", project="p", domain="d", run_name="parent-run"):
    """A Context carrying a TaskContext, as the in-container runtime would set up."""
    action = ActionID(name="a0", run_name=run_name, project=project, domain=domain, org=org)
    task_context = TaskContext(
        action=action,
        version="v1",
        raw_data_path=RawDataPath(path="/tmp/raw"),
        output_path="/tmp/out",
        run_base_dir="/tmp/base",
        report=Report(name="a0"),
        mode=mode,
    )
    return Context(data=ContextData(task_context=task_context))


@pytest.mark.asyncio
async def test_no_ctx_returns_none():
    """No in-container task context -> nothing spawned this run -> no pointer."""
    await _init_for_testing(project="p", domain="d", org="o")
    assert _Runner(force_mode="remote")._resolve_spawn_parent() is None


@pytest.mark.asyncio
async def test_remote_ctx_stamps_current_run():
    await _init_for_testing(project="p", domain="d", org="o")
    with _fake_task_ctx(mode="remote"):
        parent = _Runner(force_mode="remote")._resolve_spawn_parent()
    assert parent == identifier_pb2.RunIdentifier(org="o", project="p", domain="d", name="parent-run")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["local", "hybrid"])
async def test_non_remote_ctx_not_stamped(mode):
    """Only a true in-container (remote) execution links its child runs."""
    await _init_for_testing(project="p", domain="d", org="o")
    with _fake_task_ctx(mode=mode):
        assert _Runner(force_mode="remote")._resolve_spawn_parent() is None


@pytest.mark.asyncio
async def test_scope_mismatch_runner_project_override():
    """Spawn link is same-scope by contract: a child targeting another project is not linked."""
    await _init_for_testing(project="p", domain="d", org="o")
    with _fake_task_ctx(mode="remote"):
        assert _Runner(force_mode="remote", project="other")._resolve_spawn_parent() is None


@pytest.mark.asyncio
async def test_scope_mismatch_source_org():
    """An invoking run in a different org than the new run's target is not linked."""
    await _init_for_testing(project="p", domain="d", org="o")
    with _fake_task_ctx(mode="remote", org="different"):
        assert _Runner(force_mode="remote")._resolve_spawn_parent() is None


@pytest.mark.asyncio
async def test_empty_org_returns_none():
    """All four id fields are server-required (min_len=1); no org configured -> no pointer."""
    await _init_for_testing(project="p", domain="d")
    with _fake_task_ctx(mode="remote", org=None):
        assert _Runner(force_mode="remote")._resolve_spawn_parent() is None


@pytest.mark.asyncio
async def test_source_scope_filled_from_cfg():
    """An invoking action with empty scope fields inherits the init config's scope."""
    await _init_for_testing(project="p", domain="d", org="o")
    with _fake_task_ctx(mode="remote", org="", project="", domain=""):
        parent = _Runner(force_mode="remote")._resolve_spawn_parent()
    assert parent == identifier_pb2.RunIdentifier(org="o", project="p", domain="d", name="parent-run")
