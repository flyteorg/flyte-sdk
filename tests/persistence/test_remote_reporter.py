"""Tests for the local-run control-plane reporter (RemoteRunReporter).

Runs tasks through the local controller with a fake ClientSet injected via
``_init_for_testing`` (mirroring tests/flyte/local_controller/test_tracker_integration.py)
and asserts on the CreateRun / ReportActions / UploadMetadata traffic.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.common import identifier_pb2
from flyteidl2.dataproxy import dataproxy_service_pb2
from flyteidl2.workflow import local_run_service_pb2, run_service_pb2
from google.rpc import status_pb2

import flyte
import flyte.errors
from flyte._initialize import _init_for_testing
from flyte._persistence._remote_reporter import (
    ROOT_ACTION_NAME,
    RemoteRunReporter,
    generate_local_run_name,
    validate_local_run_name,
)
from flyte.remote._client.controlplane import Console

_PHASE_RUNNING = 4
_PHASE_SUCCEEDED = 5
_PHASE_FAILED = 6

env = flyte.TaskEnvironment(name="remote_reporter_test")

_flaky_attempts = {"count": 0}


@env.task
def add(a: int, b: int) -> int:
    return a + b


@env.task
def failing_task(x: int) -> int:
    raise ValueError("intentional failure")


@env.task
def parent_task(n: int) -> int:
    return sum(add(a=i, b=i) for i in range(n))


@env.task(retries=1)
def flaky(x: int) -> int:
    _flaky_attempts["count"] += 1
    if _flaky_attempts["count"] < 2:
        raise RuntimeError("transient error")
    return x


def _make_fake_client():
    client = MagicMock()
    client.local_run_service = MagicMock()
    client.local_run_service.create_run = AsyncMock(return_value=run_service_pb2.CreateRunResponse())

    async def _report(req, **kwargs):
        return local_run_service_pb2.ReportLocalActionsResponse(
            statuses=[status_pb2.Status(code=0) for _ in req.updates]
        )

    client.local_run_service.report_actions = AsyncMock(side_effect=_report)
    client.dataproxy_service = MagicMock()
    client.dataproxy_service.upload_metadata = AsyncMock(
        return_value=dataproxy_service_pb2.CreateUploadLocationResponse(
            signed_url="https://signed.example/put", native_url="s3://bucket/meta/artifact"
        )
    )
    client.console = Console("dns:///example.com", insecure=False)
    return client


@pytest.fixture
def fake_client():
    import flyte._initialize as init_mod

    prev = init_mod._init_config
    client = _make_fake_client()
    asyncio.run(_init_for_testing(project="testproj", domain="dev", org="testorg", client=client))
    with patch("flyte._persistence._remote_upload._put_bytes_with_retry", new=AsyncMock(return_value=None)):
        yield client
    init_mod._init_config = prev


def _all_updates(client) -> list:
    updates = []
    for call in client.local_run_service.report_actions.await_args_list:
        updates.extend(call[0][0].updates)
    return updates


def _events_for(updates, action_name):
    return [u.event for u in updates if u.event.id.name == action_name]


# ---------------------------------------------------------------------------
# End-to-end reporting through the local controller
# ---------------------------------------------------------------------------


def test_report_creates_run_and_reports_actions(fake_client):
    run = flyte.with_runcontext(mode="local", report=True).run(add, a=2, b=3)

    assert run.outputs()[0] == 5

    # CreateRun called once with a fully-qualified run id and a valid generated name.
    fake_client.local_run_service.create_run.assert_awaited_once()
    create_req = fake_client.local_run_service.create_run.await_args[0][0]
    assert create_req.run_id.org == "testorg"
    assert create_req.run_id.project == "testproj"
    assert create_req.run_id.domain == "dev"
    run_name = create_req.run_id.name
    assert len(run_name) <= 30
    assert not run_name.startswith(("u", "r"))
    assert create_req.WhichOneof("task") == "task_spec"
    assert create_req.HasField("run_start_time")

    # Root inputs were offloaded through UploadMetadata(INPUTS) targeting a0.
    assert create_req.offloaded_input_data.uri == "s3://bucket/meta/artifact"
    assert create_req.offloaded_input_data.inputs_hash != ""
    inputs_uploads = [
        c[0][0]
        for c in fake_client.dataproxy_service.upload_metadata.await_args_list
        if c[0][0].artifact_type == dataproxy_service_pb2.ARTIFACT_TYPE_INPUTS
    ]
    assert len(inputs_uploads) == 1
    assert inputs_uploads[0].WhichOneof("target") == "action_id"
    assert inputs_uploads[0].action_id.name == ROOT_ACTION_NAME
    assert inputs_uploads[0].action_id.run == create_req.run_id
    assert len(inputs_uploads[0].content_md5) == 16

    # Every reported update references the created run.
    for call in fake_client.local_run_service.report_actions.await_args_list:
        assert call[0][0].run_id == create_req.run_id

    updates = _all_updates(fake_client)

    # Root a0 mirrors record_root_*: RUNNING then SUCCEEDED, no parent.
    root_events = _events_for(updates, ROOT_ACTION_NAME)
    assert [e.phase for e in root_events] == [_PHASE_RUNNING, _PHASE_SUCCEEDED]
    root_first = next(u for u in updates if u.event.id.name == ROOT_ACTION_NAME and u.event.phase == _PHASE_RUNNING)
    assert root_first.parent_name == ""

    # The task action: RUNNING then SUCCEEDED with attempt=1, monotonic version.
    (task_action_name,) = {u.event.id.name for u in updates} - {ROOT_ACTION_NAME}
    task_events = _events_for(updates, task_action_name)
    assert [e.phase for e in task_events] == [_PHASE_RUNNING, _PHASE_SUCCEEDED]
    assert all(e.attempt == 1 for e in task_events)
    versions = [e.version for e in task_events]
    assert versions == sorted(versions)
    assert len(set(versions)) == len(versions)

    # First report carries parenting + a task spec; outputs were uploaded and referenced.
    first_update = next(u for u in updates if u.event.id.name == task_action_name and u.event.phase == _PHASE_RUNNING)
    assert first_update.parent_name == ROOT_ACTION_NAME
    assert first_update.WhichOneof("spec") == "task"
    assert first_update.task.spec.task_template.id.name == add.name
    terminal = next(e for e in task_events if e.phase == _PHASE_SUCCEEDED)
    assert terminal.outputs.output_uri == "s3://bucket/meta/artifact"
    outputs_uploads = [
        c[0][0]
        for c in fake_client.dataproxy_service.upload_metadata.await_args_list
        if c[0][0].artifact_type == dataproxy_service_pb2.ARTIFACT_TYPE_OUTPUTS
    ]
    assert len(outputs_uploads) == 1
    assert outputs_uploads[0].WhichOneof("target") == "action_attempt_id"
    assert outputs_uploads[0].action_attempt_id.attempt == 1
    assert outputs_uploads[0].action_attempt_id.action_id.name == task_action_name

    # The returned run points at the console local-runs page.
    assert run.url == f"https://example.com/v2/domain/dev/project/testproj/local-runs/{run_name}"


def test_report_nested_tasks_parent_chain(fake_client):
    flyte.with_runcontext(mode="local", report=True).run(parent_task, n=2)

    updates = _all_updates(fake_client)
    first_reports = {u.event.id.name: u for u in updates if u.WhichOneof("spec") == "task"}
    parents = {name: u.parent_name for name, u in first_reports.items()}
    # One top-level action under a0, two children under it.
    top = [name for name, parent in parents.items() if parent == ROOT_ACTION_NAME]
    assert len(top) == 1
    children = [name for name, parent in parents.items() if parent == top[0]]
    assert len(children) == 2


def test_report_failure_reports_failed(fake_client):
    with pytest.raises(Exception):
        flyte.with_runcontext(mode="local", report=True).run(failing_task, x=1)

    updates = _all_updates(fake_client)
    root_events = _events_for(updates, ROOT_ACTION_NAME)
    assert root_events[-1].phase == _PHASE_FAILED
    assert "intentional failure" in root_events[-1].error_info.message

    (task_action_name,) = {u.event.id.name for u in updates} - {ROOT_ACTION_NAME}
    task_events = _events_for(updates, task_action_name)
    failed = [e for e in task_events if e.phase == _PHASE_FAILED]
    assert failed
    assert "intentional failure" in failed[-1].error_info.message
    assert all(e.attempt == 1 for e in task_events)


def test_report_retries_report_attempts(fake_client):
    _flaky_attempts["count"] = 0
    run = flyte.with_runcontext(mode="local", report=True).run(flaky, x=7)
    assert run.outputs()[0] == 7

    updates = _all_updates(fake_client)
    (task_action_name,) = {u.event.id.name for u in updates} - {ROOT_ACTION_NAME}
    task_events = _events_for(updates, task_action_name)

    attempt1 = [e for e in task_events if e.attempt == 1]
    attempt2 = [e for e in task_events if e.attempt == 2]
    assert [e.phase for e in attempt1] == [_PHASE_RUNNING, _PHASE_FAILED]
    assert [e.phase for e in attempt2] == [_PHASE_RUNNING, _PHASE_SUCCEEDED]
    # Versions restart and stay monotonic per attempt.
    for events in (attempt1, attempt2):
        versions = [e.version for e in events]
        assert versions == sorted(versions)
        assert len(set(versions)) == len(versions)


def test_reporting_failure_does_not_fail_run(fake_client):
    fake_client.local_run_service.report_actions = AsyncMock(side_effect=RuntimeError("control plane down"))

    with patch("flyte._persistence._remote_reporter._SEND_BACKOFF_SEC", 0.01):
        run = flyte.with_runcontext(mode="local", report=True).run(add, a=1, b=1)

    assert run.outputs()[0] == 2


def test_upload_failure_does_not_fail_run(fake_client):
    fake_client.dataproxy_service.upload_metadata = AsyncMock(side_effect=RuntimeError("no storage"))

    run = flyte.with_runcontext(mode="local", report=True).run(add, a=1, b=2)

    assert run.outputs()[0] == 3
    # CreateRun proceeds without offloaded inputs? No — the inputs upload happens before
    # CreateRun, so registration fails and the run silently continues unreported.
    assert fake_client.local_run_service.report_actions.await_count == 0


def test_create_run_failure_falls_back_to_unreported(fake_client):
    fake_client.local_run_service.create_run = AsyncMock(side_effect=RuntimeError("nope"))

    run = flyte.with_runcontext(mode="local", report=True).run(add, a=4, b=4)

    assert run.outputs()[0] == 8
    assert fake_client.local_run_service.report_actions.await_count == 0
    # Falls back to the local metadata path URL.
    assert not run.url.startswith("https://")


def test_no_client_warns_and_skips():
    import flyte._initialize as init_mod

    prev = init_mod._init_config
    try:
        asyncio.run(_init_for_testing(project="testproj", domain="dev", client=None))
        run = flyte.with_runcontext(mode="local", report=True).run(add, a=1, b=5)
        assert run.outputs()[0] == 6
        assert not run.url.startswith("https://")
    finally:
        init_mod._init_config = prev


def test_missing_project_domain_raises():
    import flyte._initialize as init_mod

    prev = init_mod._init_config
    try:
        asyncio.run(_init_for_testing(client=_make_fake_client()))
        with pytest.raises(flyte.errors.InitializationError):
            flyte.with_runcontext(mode="local", report=True).run(add, a=1, b=1)
    finally:
        init_mod._init_config = prev


def test_report_only_valid_in_local_mode(fake_client):
    with pytest.raises(ValueError, match="only supported in local mode"):
        flyte.with_runcontext(mode="remote", report=True).run(add, a=1, b=1)


# ---------------------------------------------------------------------------
# Run-name contract
# ---------------------------------------------------------------------------


def test_generated_run_name_is_compliant():
    for _ in range(50):
        name = generate_local_run_name()
        assert len(name) <= 30
        assert not name.startswith(("u", "r"))


def test_validate_run_name_rejects_reserved_prefixes():
    with pytest.raises(ValueError, match="reserved"):
        validate_local_run_name("uplifting-run")
    with pytest.raises(ValueError, match="reserved"):
        validate_local_run_name("racy-run")
    with pytest.raises(ValueError, match="too long"):
        validate_local_run_name("x" * 31)
    validate_local_run_name("my-local-run")


def test_invalid_user_run_name_fails_fast(fake_client):
    with pytest.raises(ValueError, match="reserved"):
        flyte.with_runcontext(mode="local", report=True, name="urgent").run(add, a=1, b=1)
    fake_client.local_run_service.create_run.assert_not_awaited()


# ---------------------------------------------------------------------------
# Reporter unit behavior (direct, no controller)
# ---------------------------------------------------------------------------


def _make_reporter(client=None, **kwargs) -> RemoteRunReporter:
    run_id = identifier_pb2.RunIdentifier(org="o", project="p", domain="d", name="local-abc")
    return RemoteRunReporter(client or _make_fake_client(), run_id, **kwargs)


def test_flush_barrier_is_bounded():
    client = _make_fake_client()

    async def _hang(req, **kwargs):
        await asyncio.sleep(60)

    client.local_run_service.report_actions = AsyncMock(side_effect=_hang)
    reporter = _make_reporter(client, flush_timeout_sec=0.5)
    reporter.record_root_start(task_name="t")

    start = time.monotonic()
    reporter.close()
    assert time.monotonic() - start < 5


def test_terminal_dedupe_between_attempt_complete_and_complete():
    reporter = _make_reporter()
    events = []
    reporter._put = events.append  # intercept before the worker consumes anything

    reporter.record_start(action_id="a1", task_name="t")
    reporter.record_attempt_start(action_id="a1", attempt_num=1)
    reporter.record_attempt_complete(action_id="a1", attempt_num=1)
    reporter.record_complete(action_id="a1")

    phases = [e.phase for e in events]
    assert phases == [_PHASE_RUNNING, _PHASE_SUCCEEDED]
    versions = [e.version for e in events]
    assert versions == [0, 1]
    reporter.close(timeout=1)


def test_enqueue_never_raises_after_close():
    reporter = _make_reporter()
    reporter.close(timeout=1)
    # Late events are dropped silently.
    reporter.record_start(action_id="a1", task_name="t")
    reporter.record_complete(action_id="a1")
