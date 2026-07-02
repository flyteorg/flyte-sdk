"""Unit tests for _Runner._resolve_related_to: the provenance pointer (RunSpec.related_to)
resolution rules. These run on any flyteidl2 build — the resolver only constructs a
RunIdentifier; the descriptor-gated stamping is covered in test_run_runspec_chars.py."""

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
async def test_no_ctx_no_source_returns_none():
    await _init_for_testing(project="p", domain="d", org="o")
    assert _Runner(force_mode="remote")._resolve_related_to() is None


@pytest.mark.asyncio
async def test_remote_ctx_stamps_current_run():
    await _init_for_testing(project="p", domain="d", org="o")
    with _fake_task_ctx(mode="remote"):
        related_to = _Runner(force_mode="remote")._resolve_related_to()
    assert related_to == identifier_pb2.RunIdentifier(org="o", project="p", domain="d", name="parent-run")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["local", "hybrid"])
async def test_non_remote_ctx_not_stamped(mode):
    """Only a true in-container (remote) execution links its child runs."""
    await _init_for_testing(project="p", domain="d", org="o")
    with _fake_task_ctx(mode=mode):
        assert _Runner(force_mode="remote")._resolve_related_to() is None


@pytest.mark.asyncio
async def test_explicit_source_wins_over_ctx():
    """The rerun path passes the source run explicitly; ambient ctx must not leak in."""
    await _init_for_testing(project="p", domain="d", org="o")
    source = identifier_pb2.RunIdentifier(org="o", project="p", domain="d", name="src-run")
    with _fake_task_ctx(mode="remote", run_name="ctx-run"):
        related_to = _Runner(force_mode="remote")._resolve_related_to(source)
    assert related_to is not None
    assert related_to.name == "src-run"


@pytest.mark.asyncio
async def test_scope_mismatch_runner_project_override():
    """related_to is same-scope by contract: a run targeting another project is not linked."""
    await _init_for_testing(project="p", domain="d", org="o")
    source = identifier_pb2.RunIdentifier(org="o", project="p", domain="d", name="src-run")
    assert _Runner(force_mode="remote", project="other")._resolve_related_to(source) is None


@pytest.mark.asyncio
async def test_scope_mismatch_source_org():
    await _init_for_testing(project="p", domain="d", org="o")
    source = identifier_pb2.RunIdentifier(org="different", project="p", domain="d", name="src-run")
    assert _Runner(force_mode="remote")._resolve_related_to(source) is None


@pytest.mark.asyncio
async def test_empty_org_returns_none():
    """All four id fields are server-required (min_len=1); no org configured -> no pointer."""
    await _init_for_testing(project="p", domain="d")
    with _fake_task_ctx(mode="remote", org=None):
        assert _Runner(force_mode="remote")._resolve_related_to() is None


@pytest.mark.asyncio
async def test_source_scope_filled_from_cfg():
    """A name-only source (degenerate fetched id) inherits the init config's scope."""
    await _init_for_testing(project="p", domain="d", org="o")
    source = identifier_pb2.RunIdentifier(name="src-run")
    related_to = _Runner(force_mode="remote")._resolve_related_to(source)
    assert related_to == identifier_pb2.RunIdentifier(org="o", project="p", domain="d", name="src-run")
