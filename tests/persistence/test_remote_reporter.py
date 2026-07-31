"""Tests for the local-run control-plane reporter (RemoteRunReporter).

Runs tasks through the local controller with a fake ClientSet injected via
``_init_for_testing`` (mirroring tests/flyte/local_controller/test_tracker_integration.py)
and asserts on the CreateRun / ReportActions / UploadMetadata traffic.

The fake backend is stateful and mirrors the real server contract: UploadMetadata for
OUTPUTS / REPORT fails with "missing entity" unless the target action was already
created by an acked report (or CreateRun for a0), and the signed-PUT payloads are
captured so tests assert on the actual serialized Inputs/Outputs proto bytes.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.common import identifier_pb2
from flyteidl2.dataproxy import dataproxy_service_pb2
from flyteidl2.task import common_pb2 as task_common_pb2
from flyteidl2.workflow import local_run_service_pb2, run_service_pb2
from google.rpc import status_pb2

import flyte
import flyte.errors
from flyte._initialize import _init_for_testing
from flyte._persistence._remote_reporter import (
    ROOT_ACTION_NAME,
    LocalRunReportingError,
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


@env.task(report=True)
async def report_task(x: int) -> int:
    import flyte.report

    flyte.report.get_tab("t").log("<p>report-marker</p>")
    await flyte.report.flush.aio()
    return x


@flyte.trace
def traced_double(v: int) -> int:
    return v * 2


@env.task
def task_with_trace(x: int) -> int:
    return traced_double(x)


def _make_fake_client():
    """A stateful fake ClientSet mirroring the server's UploadMetadata contract.

    - ``create_run`` creates the root action a0 (as the real server does).
    - ``report_actions`` creates every reported action on first ack.
    - ``upload_metadata`` for OUTPUTS / REPORT fails with "missing entity" unless the
      target action already exists — so a regression that uploads before the action's
      first report is acked fails these tests, not just the live path.

    Uploads are recorded as ``(artifact_type, action, attempt, md5-hex)`` in
    ``client.uploads``; the fixture pairs them with the actual PUT payload bytes in
    ``client.put_payloads`` keyed by md5-hex.
    """
    client = MagicMock()
    client.reported_actions = set()
    client.uploads = []
    client.put_payloads = {}

    client.local_run_service = MagicMock()

    async def _create_run(req, **kwargs):
        client.reported_actions.add(ROOT_ACTION_NAME)
        return run_service_pb2.CreateRunResponse()

    client.local_run_service.create_run = AsyncMock(side_effect=_create_run)

    async def _report(req, **kwargs):
        for u in req.updates:
            client.reported_actions.add(u.event.id.name)
        return local_run_service_pb2.ReportLocalActionsResponse(
            statuses=[status_pb2.Status(code=0) for _ in req.updates]
        )

    client.local_run_service.report_actions = AsyncMock(side_effect=_report)

    client.dataproxy_service = MagicMock()

    async def _upload_metadata(req, **kwargs):
        if req.artifact_type == dataproxy_service_pb2.ARTIFACT_TYPE_INPUTS:
            action, attempt = req.action_id.name, 0
            suffix = "inputs.pb"
        else:
            action = req.action_attempt_id.action_id.name
            attempt = req.action_attempt_id.attempt
            # Server contract: outputs/reports are uploaded after the action was
            # reported, so the action must already exist.
            if action not in client.reported_actions:
                raise RuntimeError(f"missing entity of type LocalAction with identifier {action}")
            suffix = "outputs.pb" if req.artifact_type == dataproxy_service_pb2.ARTIFACT_TYPE_OUTPUTS else "report.html"
        client.uploads.append((req.artifact_type, action, attempt, req.content_md5.hex()))
        return dataproxy_service_pb2.CreateUploadLocationResponse(
            signed_url="https://signed.example/put",
            native_url=f"s3://bucket/meta/{action}/{attempt}/{suffix}",
        )

    client.dataproxy_service.upload_metadata = AsyncMock(side_effect=_upload_metadata)
    client.console = Console("dns:///example.com", insecure=False)
    return client


def _uploaded_bytes(client, artifact_type: int, action: str, attempt: int | None = None) -> bytes | None:
    """The actual PUT payload for the recorded upload, or None when never uploaded."""
    for t, a, att, md5_hex in client.uploads:
        if t == artifact_type and a == action and (attempt is None or att == attempt):
            return client.put_payloads.get(md5_hex)
    return None


@pytest.fixture
def fake_client():
    import flyte._initialize as init_mod

    prev = init_mod._init_config
    client = _make_fake_client()
    asyncio.run(_init_for_testing(project="testproj", domain="dev", org="testorg", client=client))

    async def _capture_put(data, **kwargs):
        client.put_payloads[hashlib.md5(data).hexdigest()] = bytes(data)

    with patch("flyte._persistence._remote_upload._put_bytes_with_retry", new=AsyncMock(side_effect=_capture_put)):
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

    # Root inputs were offloaded through UploadMetadata(INPUTS) targeting a0, and the
    # uploaded payload is the actual serialized Inputs proto (a=2, b=3).
    assert create_req.offloaded_input_data.uri == f"s3://bucket/meta/{ROOT_ACTION_NAME}/0/inputs.pb"
    assert create_req.offloaded_input_data.inputs_hash != ""
    root_inputs_bytes = _uploaded_bytes(fake_client, dataproxy_service_pb2.ARTIFACT_TYPE_INPUTS, ROOT_ACTION_NAME)
    assert root_inputs_bytes is not None
    root_inputs = task_common_pb2.Inputs.FromString(root_inputs_bytes)
    assert {nl.name: nl.value.scalar.primitive.integer for nl in root_inputs.literals} == {"a": 2, "b": 3}

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

    # CreateRun's root task spec carries the typed interface.
    root_iface = create_req.task_spec.task_template.interface
    assert {e.key for e in root_iface.inputs.variables} == {"a", "b"}
    assert {e.key for e in root_iface.outputs.variables} == {"o0"}

    # First report carries parenting + a FULL task spec including the typed interface
    # (the console gates I/O rendering on it — identifier-only specs must not regress).
    first_update = next(u for u in updates if u.event.id.name == task_action_name and u.event.phase == _PHASE_RUNNING)
    assert first_update.parent_name == ROOT_ACTION_NAME
    assert first_update.WhichOneof("spec") == "task"
    assert first_update.task.spec.task_template.id.name == add.name
    child_iface = first_update.task.spec.task_template.interface
    assert {e.key for e in child_iface.inputs.variables} == {"a", "b"}
    assert {e.key for e in child_iface.outputs.variables} == {"o0"}

    # The child action's inputs.pb was uploaded and carries the real Inputs proto.
    child_inputs_bytes = _uploaded_bytes(fake_client, dataproxy_service_pb2.ARTIFACT_TYPE_INPUTS, task_action_name)
    assert child_inputs_bytes is not None
    child_inputs = task_common_pb2.Inputs.FromString(child_inputs_bytes)
    assert {nl.name: nl.value.scalar.primitive.integer for nl in child_inputs.literals} == {"a": 2, "b": 3}

    # The succeeded attempt uploaded outputs.pb (real Outputs proto: o0 = 5) and the
    # terminal event references it. The stateful fake rejects uploads for actions the
    # server hasn't seen, so this also proves the report-then-upload ordering.
    terminal = next(e for e in task_events if e.phase == _PHASE_SUCCEEDED)
    assert terminal.outputs.output_uri == f"s3://bucket/meta/{task_action_name}/1/outputs.pb"
    child_outputs_bytes = _uploaded_bytes(
        fake_client, dataproxy_service_pb2.ARTIFACT_TYPE_OUTPUTS, task_action_name, attempt=1
    )
    assert child_outputs_bytes is not None
    child_outputs = task_common_pb2.Outputs.FromString(child_outputs_bytes)
    assert {nl.name: nl.value.scalar.primitive.integer for nl in child_outputs.literals} == {"o0": 5}

    # record_root_complete replicated the driver outputs for a0's attempt.
    root_outputs_bytes = _uploaded_bytes(fake_client, dataproxy_service_pb2.ARTIFACT_TYPE_OUTPUTS, ROOT_ACTION_NAME, 1)
    assert root_outputs_bytes == child_outputs_bytes
    root_terminal = next(e for e in root_events if e.phase == _PHASE_SUCCEEDED)
    assert root_terminal.outputs.output_uri == f"s3://bucket/meta/{ROOT_ACTION_NAME}/1/outputs.pb"

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

    # Every first-report spec (top-level and children) carries a typed interface.
    for update in first_reports.values():
        iface = update.task.spec.task_template.interface
        assert len(iface.inputs.variables) > 0
        assert len(iface.outputs.variables) > 0


def test_report_trace_action_spec_carries_interface(fake_client):
    """@flyte.trace pseudo-actions are reported with an interface-bearing spec.

    They are sent as task actions: the local-run backend currently discards
    TraceAction.spec (only the action type is recorded), so a TraceSpec cannot
    surface the typed interface the console's I/O panels are gated on.
    """
    run = flyte.with_runcontext(mode="local", report=True).run(task_with_trace, x=3)
    assert run.outputs()[0] == 6

    updates = _all_updates(fake_client)
    trace_updates = [
        u for u in updates if u.WhichOneof("spec") == "task" and u.task.spec.task_template.id.name == "traced_double"
    ]
    assert len(trace_updates) == 1
    iface = trace_updates[0].task.spec.task_template.interface
    assert {e.key for e in iface.inputs.variables} == {"v"}
    assert len(iface.outputs.variables) > 0


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

    # The succeeded attempt (2) uploaded its outputs and referenced them.
    outputs_bytes = _uploaded_bytes(fake_client, dataproxy_service_pb2.ARTIFACT_TYPE_OUTPUTS, task_action_name, 2)
    assert outputs_bytes is not None
    outputs = task_common_pb2.Outputs.FromString(outputs_bytes)
    assert {nl.name: nl.value.scalar.primitive.integer for nl in outputs.literals} == {"o0": 7}
    succeeded = next(e for e in attempt2 if e.phase == _PHASE_SUCCEEDED)
    assert succeeded.outputs.output_uri == f"s3://bucket/meta/{task_action_name}/2/outputs.pb"


def test_report_html_uploaded_and_referenced(fake_client):
    run = flyte.with_runcontext(mode="local", report=True).run(report_task, x=9)
    assert run.outputs()[0] == 9

    updates = _all_updates(fake_client)
    (task_action_name,) = {u.event.id.name for u in updates} - {ROOT_ACTION_NAME}

    report_bytes = _uploaded_bytes(fake_client, dataproxy_service_pb2.ARTIFACT_TYPE_REPORT, task_action_name, 1)
    assert report_bytes is not None
    assert b"report-marker" in report_bytes

    terminal = next(e for e in _events_for(updates, task_action_name) if e.phase == _PHASE_SUCCEEDED)
    assert terminal.outputs.report_uri == f"s3://bucket/meta/{task_action_name}/1/report.html"
    assert terminal.outputs.output_uri == f"s3://bucket/meta/{task_action_name}/1/outputs.pb"


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
# Strict reporting mode (report_strict): every failure class fails the run loudly
# ---------------------------------------------------------------------------


def _fail_terminal_uploads(client):
    """Make OUTPUTS/REPORT uploads fail while INPUTS (and thus bootstrap) succeed."""
    orig = client.dataproxy_service.upload_metadata.side_effect

    async def _upload(req, **kwargs):
        if req.artifact_type != dataproxy_service_pb2.ARTIFACT_TYPE_INPUTS:
            raise RuntimeError("upload rejected")
        return await orig(req, **kwargs)

    client.dataproxy_service.upload_metadata = AsyncMock(side_effect=_upload)


def test_strict_bootstrap_failure_fails_run(fake_client):
    fake_client.local_run_service.create_run = AsyncMock(side_effect=RuntimeError("nope"))

    with pytest.raises(LocalRunReportingError, match="register local run"):
        flyte.with_runcontext(mode="local", report=True, report_strict=True).run(add, a=1, b=1)


def test_strict_upload_failure_fails_run(fake_client):
    _fail_terminal_uploads(fake_client)

    with pytest.raises(LocalRunReportingError, match="outputs upload"):
        flyte.with_runcontext(mode="local", report=True, report_strict=True).run(add, a=1, b=1)


def test_default_mode_upload_failure_does_not_fail_run(fake_client):
    """Counterpart: the same terminal-upload failure is swallowed under the default policy."""
    _fail_terminal_uploads(fake_client)

    run = flyte.with_runcontext(mode="local", report=True).run(add, a=1, b=1)
    assert run.outputs()[0] == 2


def test_strict_transport_failure_fails_run(fake_client):
    fake_client.local_run_service.report_actions = AsyncMock(side_effect=RuntimeError("control plane down"))

    with patch("flyte._persistence._remote_reporter._SEND_BACKOFF_SEC", 0.01):
        with pytest.raises(LocalRunReportingError, match="ReportActions"):
            flyte.with_runcontext(mode="local", report=True, report_strict=True).run(add, a=1, b=1)


def test_strict_rejected_update_fails_run(fake_client):
    async def _reject(req, **kwargs):
        return local_run_service_pb2.ReportLocalActionsResponse(
            statuses=[status_pb2.Status(code=3, message="validation failed") for _ in req.updates]
        )

    fake_client.local_run_service.report_actions = AsyncMock(side_effect=_reject)

    with pytest.raises(LocalRunReportingError, match="validation failed"):
        flyte.with_runcontext(mode="local", report=True, report_strict=True).run(add, a=1, b=1)


def test_default_mode_rejected_update_does_not_fail_run(fake_client):
    async def _reject(req, **kwargs):
        return local_run_service_pb2.ReportLocalActionsResponse(
            statuses=[status_pb2.Status(code=3, message="validation failed") for _ in req.updates]
        )

    fake_client.local_run_service.report_actions = AsyncMock(side_effect=_reject)

    run = flyte.with_runcontext(mode="local", report=True).run(add, a=1, b=1)
    assert run.outputs()[0] == 2


def test_strict_flush_timeout_raises():
    client = _make_fake_client()

    async def _hang(req, **kwargs):
        await asyncio.sleep(60)

    client.local_run_service.report_actions = AsyncMock(side_effect=_hang)
    reporter = _make_reporter(client, flush_timeout_sec=0.3, strict=True)
    reporter.record_root_start(task_name="t")

    start = time.monotonic()
    with pytest.raises(LocalRunReportingError, match="Timed out"):
        reporter.close()
    assert time.monotonic() - start < 5


def test_strict_enqueue_reraises_first_failure():
    reporter = _make_reporter(strict=True)
    reporter._note_failure("outputs upload", "a1", "boom")

    with pytest.raises(LocalRunReportingError, match="outputs upload"):
        reporter.record_start(action_id="a2", task_name="t")
    # The failing path never fast-raises so it cannot mask the task's own error.
    reporter.record_failure(action_id="a2", error="task error")
    with pytest.raises(LocalRunReportingError):
        reporter.close(timeout=1)


def test_strict_requires_report_enabled(fake_client):
    with pytest.raises(ValueError, match="requires reporting to be enabled"):
        flyte.with_runcontext(mode="local", report_strict=True).run(add, a=1, b=1)


def test_strict_requires_client():
    import flyte._initialize as init_mod

    prev = init_mod._init_config
    try:
        asyncio.run(_init_for_testing(project="testproj", domain="dev", client=None))
        with pytest.raises(flyte.errors.InitializationError):
            flyte.with_runcontext(mode="local", report=True, report_strict=True).run(add, a=1, b=1)
    finally:
        init_mod._init_config = prev


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
