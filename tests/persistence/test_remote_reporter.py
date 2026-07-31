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
from flyte.io import File
from flyte.remote._client.controlplane import Console

_PHASE_RUNNING = 4
_PHASE_SUCCEEDED = 5
_PHASE_FAILED = 6
_PHASE_ABORTED = 7

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


@env.task
def empty_str_task() -> str:
    return ""


@env.task(report=True)
async def multi_flush_report_task(x: int) -> int:
    import asyncio as _asyncio

    import flyte.report

    flyte.report.get_tab("t").log("<p>flush-one</p>")
    await flyte.report.flush.aio()
    # Give the reporter worker time to drain the first snapshot as its own batch.
    await _asyncio.sleep(0.3)
    flyte.report.get_tab("t").log("<p>flush-two</p>")
    await flyte.report.flush.aio()
    return x


# A real Ctrl+C under asyncio.run delivers KeyboardInterrupt to the main thread,
# which cancels the running task — surfacing at _run_local's await as
# CancelledError. Raising CancelledError from task code reproduces exactly that
# boundary (raising KeyboardInterrupt inside a worker loop instead tears the loop
# down before its future resolves, which is not what Ctrl+C does).
@env.task
async def interrupt_child(x: int) -> int:
    raise asyncio.CancelledError


@env.task
async def interrupt_parent(n: int) -> int:
    return await interrupt_child(x=n)


@env.task
async def file_roundtrip(f: "File") -> "File":
    return f


@env.task
async def file_producer() -> "File":
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
        tmp.write("produced-locally")
        path = tmp.name
    return await File.from_local(path)


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
    client.put_headers = {}

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
        md5_hex = hashlib.md5(data).hexdigest()
        client.put_payloads[md5_hex] = bytes(data)
        client.put_headers[md5_hex] = dict(kwargs.get("extra_headers") or {})

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
    """A single-task run: the driver action is mapped onto the root a0 (platform
    semantics — the root action IS the driver execution, no duplicate)."""
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

    # The driver is mapped onto a0 — no random-id duplicate action is ever reported.
    assert {u.event.id.name for u in updates} == {ROOT_ACTION_NAME}

    # a0's event stream merges root-start and driver events: RUNNING (root), RUNNING
    # (driver start), SUCCEEDED (driver terminal) — attempt 1 with a single shared,
    # monotonic version counter, and exactly one terminal event (root synthesis is
    # deduped by the driver's own terminal).
    root_events = _events_for(updates, ROOT_ACTION_NAME)
    assert [e.phase for e in root_events] == [_PHASE_RUNNING, _PHASE_RUNNING, _PHASE_SUCCEEDED]
    assert all(e.attempt == 1 for e in root_events)
    assert [e.version for e in root_events] == [0, 1, 2]
    root_first = next(u for u in updates if u.event.id.name == ROOT_ACTION_NAME)
    assert root_first.parent_name == ""

    # CreateRun's root task spec carries the typed interface (the console gates I/O
    # rendering on it); mapped a0 updates don't need to re-send it.
    root_iface = create_req.task_spec.task_template.interface
    assert {e.key for e in root_iface.inputs.variables} == {"a", "b"}
    assert {e.key for e in root_iface.outputs.variables} == {"o0"}

    # Exactly one INPUTS upload (the bootstrap offload for a0) — the mapped driver
    # does not re-upload under a random id.
    inputs_uploads = [u for u in fake_client.uploads if u[0] == dataproxy_service_pb2.ARTIFACT_TYPE_INPUTS]
    assert [(a, att) for _, a, att, _ in inputs_uploads] == [(ROOT_ACTION_NAME, 0)]

    # The succeeded attempt uploaded outputs.pb under a0 (real Outputs proto: o0 = 5)
    # and the terminal event references it. The stateful fake rejects uploads for
    # actions the server hasn't seen, so this also proves report-then-upload ordering.
    outputs_uploads = [u for u in fake_client.uploads if u[0] == dataproxy_service_pb2.ARTIFACT_TYPE_OUTPUTS]
    assert [(a, att) for _, a, att, _ in outputs_uploads] == [(ROOT_ACTION_NAME, 1)]
    root_outputs_bytes = _uploaded_bytes(fake_client, dataproxy_service_pb2.ARTIFACT_TYPE_OUTPUTS, ROOT_ACTION_NAME, 1)
    assert root_outputs_bytes is not None
    root_outputs = task_common_pb2.Outputs.FromString(root_outputs_bytes)
    assert {nl.name: nl.value.scalar.primitive.integer for nl in root_outputs.literals} == {"o0": 5}
    terminal = next(e for e in root_events if e.phase == _PHASE_SUCCEEDED)
    assert terminal.outputs.output_uri == f"s3://bucket/meta/{ROOT_ACTION_NAME}/1/outputs.pb"

    # The returned run points at the console local-runs page.
    assert run.url == f"https://example.com/v2/domain/dev/project/testproj/local-runs/{run_name}"


def test_report_nested_tasks_parent_chain(fake_client):
    """Fanout lineage: the driver maps onto a0, children report parent == a0 —
    never the driver's local random id (test_task_action_lineage-style guard)."""
    flyte.with_runcontext(mode="local", report=True).run(parent_task, n=2)

    updates = _all_updates(fake_client)
    first_reports = {u.event.id.name: u for u in updates if u.WhichOneof("spec") == "task"}
    # Spec-bearing first reports are the two children only (a0's spec came via CreateRun).
    assert len(first_reports) == 2
    assert all(u.parent_name == ROOT_ACTION_NAME for u in first_reports.values())

    # The whole tree is a0 + the two children — no random-id driver action anywhere,
    # and no update ever references a non-a0 parent.
    assert {u.event.id.name for u in updates} == {ROOT_ACTION_NAME} | set(first_reports)
    assert {u.parent_name for u in updates if u.parent_name} == {ROOT_ACTION_NAME}

    # a0 (the driver) delivers exactly one terminal event.
    root_terminals = [e for e in _events_for(updates, ROOT_ACTION_NAME) if e.phase == _PHASE_SUCCEEDED]
    assert len(root_terminals) == 1

    # Every child first-report spec carries a typed interface, and each child's
    # inputs.pb was uploaded under its own id.
    for name, update in first_reports.items():
        iface = update.task.spec.task_template.interface
        assert len(iface.inputs.variables) > 0
        assert len(iface.outputs.variables) > 0
        assert _uploaded_bytes(fake_client, dataproxy_service_pb2.ARTIFACT_TYPE_INPUTS, name) is not None


def test_report_trace_action_spec_carries_interface(fake_client):
    """@flyte.trace pseudo-actions report via the trace oneof: a TraceAction carrying
    the function name and a TraceSpec with the typed interface (the backend persists
    and serves trace specs)."""
    run = flyte.with_runcontext(mode="local", report=True).run(task_with_trace, x=3)
    assert run.outputs()[0] == 6

    updates = _all_updates(fake_client)
    trace_updates = [u for u in updates if u.WhichOneof("spec") == "trace"]
    assert len(trace_updates) == 1
    trace = trace_updates[0].trace
    assert trace.name == "traced_double"
    iface = trace.spec.interface
    assert {e.key for e in iface.inputs.variables} == {"v"}
    assert len(iface.outputs.variables) > 0
    # The trace nests under the driver, which is mapped onto a0.
    assert trace_updates[0].parent_name == ROOT_ACTION_NAME


def test_falsy_outputs_are_reported(fake_client):
    """Falsy results (0 here) are real outputs: uploaded and referenced based on
    presence, not truthiness — for both the trace and the task (driver) paths."""
    run = flyte.with_runcontext(mode="local", report=True).run(task_with_trace, x=0)
    assert run.outputs()[0] == 0

    updates = _all_updates(fake_client)
    (trace_name,) = {u.event.id.name for u in updates if u.WhichOneof("spec") == "trace"}

    # The x=0 trace uploaded outputs.pb carrying o0 == 0 and referenced it.
    trace_out = _uploaded_bytes(fake_client, dataproxy_service_pb2.ARTIFACT_TYPE_OUTPUTS, trace_name, 1)
    assert trace_out is not None
    outs = task_common_pb2.Outputs.FromString(trace_out)
    assert outs.literals[0].name == "o0"
    assert outs.literals[0].value.scalar.primitive.WhichOneof("value") == "integer"
    assert outs.literals[0].value.scalar.primitive.integer == 0
    terminal = next(e for e in _events_for(updates, trace_name) if e.phase == _PHASE_SUCCEEDED)
    assert terminal.outputs.output_uri.endswith(f"{trace_name}/1/outputs.pb")

    # The driver task (mapped onto a0) returning 0 uploads o0 == 0 as well.
    root_out = _uploaded_bytes(fake_client, dataproxy_service_pb2.ARTIFACT_TYPE_OUTPUTS, ROOT_ACTION_NAME, 1)
    assert root_out is not None
    root_outs = task_common_pb2.Outputs.FromString(root_out)
    assert root_outs.literals[0].value.scalar.primitive.integer == 0


def test_empty_string_output_reported(fake_client):
    """A task returning "" still uploads outputs.pb with the o0 literal present."""
    run = flyte.with_runcontext(mode="local", report=True).run(empty_str_task)
    assert run.outputs()[0] == ""

    root_out = _uploaded_bytes(fake_client, dataproxy_service_pb2.ARTIFACT_TYPE_OUTPUTS, ROOT_ACTION_NAME, 1)
    assert root_out is not None
    outs = task_common_pb2.Outputs.FromString(root_out)
    assert outs.literals[0].name == "o0"
    assert outs.literals[0].value.scalar.primitive.WhichOneof("value") == "string_value"
    assert outs.literals[0].value.scalar.primitive.string_value == ""


def test_cache_hit_reports_outputs_and_cache_status(fake_client):
    """Both the cache-miss and the cache-hit run report full trees with uploaded
    outputs, and events/status carry core.CatalogCacheStatus (MISS -> POPULATED on
    the storing run, HIT on the cached run)."""
    import random

    cache_env = flyte.TaskEnvironment(name="reporter_cache_test", cache=flyte.Cache("auto", version_override="v1"))

    @cache_env.task
    def cached_add(a: int, b: int) -> int:
        return a + b

    a = random.randint(10**6, 10**8)  # unique inputs so run 1 is always a miss
    run1 = flyte.with_runcontext(mode="local", report=True).run(cached_add, a=a, b=3)
    run2 = flyte.with_runcontext(mode="local", report=True).run(cached_add, a=a, b=3)
    assert run1.outputs()[0] == run2.outputs()[0] == a + 3

    per_run: dict[str, list] = {}
    for call in fake_client.local_run_service.report_actions.await_args_list:
        req = call[0][0]
        per_run.setdefault(req.run_id.name, []).extend(req.updates)
    assert len(per_run) == 2
    r1_updates, r2_updates = list(per_run.values())

    # Run 1 (miss): driver RUNNING carries CACHE_MISS; the storing terminal carries
    # CACHE_POPULATED on both the event and the status rollup.
    r1 = [u.event for u in r1_updates]
    assert [e.phase for e in r1] == [_PHASE_RUNNING, _PHASE_RUNNING, _PHASE_SUCCEEDED]
    assert r1[1].cache_status == 1  # CACHE_MISS
    assert r1[2].cache_status == 3  # CACHE_POPULATED
    assert r1_updates[2].status.cache_status == 3
    assert r1[2].outputs.output_uri.endswith("a0/1/outputs.pb")

    # Run 2 (hit): no attempt executed, but the tree is complete, outputs are
    # uploaded, and CACHE_HIT is carried on the running + terminal events.
    r2 = [u.event for u in r2_updates]
    assert [e.phase for e in r2] == [_PHASE_RUNNING, _PHASE_RUNNING, _PHASE_SUCCEEDED]
    assert r2[1].cache_status == 2  # CACHE_HIT
    assert r2[2].cache_status == 2
    assert r2_updates[2].status.cache_status == 2
    assert r2[2].outputs.output_uri.endswith("a0/1/outputs.pb")

    # Outputs were uploaded by BOTH runs (the hit run serves cache-sourced outputs).
    outputs_uploads = [u for u in fake_client.uploads if u[0] == dataproxy_service_pb2.ARTIFACT_TYPE_OUTPUTS]
    assert [(a_, att) for _, a_, att, _ in outputs_uploads] == [(ROOT_ACTION_NAME, 1), (ROOT_ACTION_NAME, 1)]
    payload = fake_client.put_payloads[outputs_uploads[1][3]]
    outs = task_common_pb2.Outputs.FromString(payload)
    assert outs.literals[0].value.scalar.primitive.integer == a + 3


def test_file_output_keeps_local_uri(fake_client):
    """A task RETURNING a File reports an outputs.pb whose blob literal keeps its
    local URI — no rewrite, no raw-byte upload."""
    run = flyte.with_runcontext(mode="local", report=True).run(file_producer)
    out_path = str(run.outputs()[0].path)
    assert not out_path.startswith(("s3://", "gs://", "abfs://"))

    root_out = _uploaded_bytes(fake_client, dataproxy_service_pb2.ARTIFACT_TYPE_OUTPUTS, ROOT_ACTION_NAME, 1)
    assert root_out is not None
    outs = task_common_pb2.Outputs.FromString(root_out)
    uri = outs.literals[0].value.scalar.blob.uri
    assert not uri.startswith(("s3://", "gs://", "abfs://"))
    assert out_path in uri or uri in out_path

    # Only metadata artifact kinds ever uploaded; every PUT maps to an UploadMetadata.
    kinds = {t for t, *_ in fake_client.uploads}
    assert kinds <= {
        dataproxy_service_pb2.ARTIFACT_TYPE_INPUTS,
        dataproxy_service_pb2.ARTIFACT_TYPE_OUTPUTS,
        dataproxy_service_pb2.ARTIFACT_TYPE_REPORT,
    }
    assert set(fake_client.put_payloads) == {md5 for *_, md5 in fake_client.uploads}


def test_report_upload_sets_html_content_type(fake_client):
    """report.html PUTs carry Content-Type: text/html (object metadata, so the console
    iframe renders instead of downloading); inputs/outputs stay default."""
    flyte.with_runcontext(mode="local", report=True).run(report_task, x=2)

    by_kind: dict[int, list[dict]] = {}
    for t, _a, _att, md5_hex in fake_client.uploads:
        by_kind.setdefault(t, []).append(fake_client.put_headers[md5_hex])

    report_headers = by_kind.get(dataproxy_service_pb2.ARTIFACT_TYPE_REPORT, [])
    assert report_headers
    assert all(h.get("Content-Type") == "text/html" for h in report_headers)
    for kind in (dataproxy_service_pb2.ARTIFACT_TYPE_INPUTS, dataproxy_service_pb2.ARTIFACT_TYPE_OUTPUTS):
        for h in by_kind.get(kind, []):
            assert "Content-Type" not in h


def test_report_failure_reports_failed(fake_client):
    with pytest.raises(Exception):
        flyte.with_runcontext(mode="local", report=True).run(failing_task, x=1)

    updates = _all_updates(fake_client)
    # The failing driver is mapped onto a0 — no duplicate action, and exactly one
    # FAILED terminal (record_root_failure is deduped by the driver's own failure).
    assert {u.event.id.name for u in updates} == {ROOT_ACTION_NAME}
    root_events = _events_for(updates, ROOT_ACTION_NAME)
    failed = [e for e in root_events if e.phase == _PHASE_FAILED]
    assert len(failed) == 1
    assert root_events[-1].phase == _PHASE_FAILED
    assert "intentional failure" in failed[0].error_info.message
    assert all(e.attempt == 1 for e in root_events)


def test_report_retries_report_attempts(fake_client):
    _flaky_attempts["count"] = 0
    run = flyte.with_runcontext(mode="local", report=True).run(flaky, x=7)
    assert run.outputs()[0] == 7

    updates = _all_updates(fake_client)
    # The retrying driver is mapped onto a0.
    assert {u.event.id.name for u in updates} == {ROOT_ACTION_NAME}
    task_events = _events_for(updates, ROOT_ACTION_NAME)

    # Attempt 1 merges root-start and driver-start RUNNING events, then fails;
    # attempt 2 runs and succeeds.
    attempt1 = [e for e in task_events if e.attempt == 1]
    attempt2 = [e for e in task_events if e.attempt == 2]
    assert [e.phase for e in attempt1] == [_PHASE_RUNNING, _PHASE_RUNNING, _PHASE_FAILED]
    assert [e.phase for e in attempt2] == [_PHASE_RUNNING, _PHASE_SUCCEEDED]
    # Versions restart and stay monotonic per attempt (single shared a0 counter).
    for events in (attempt1, attempt2):
        versions = [e.version for e in events]
        assert versions == sorted(versions)
        assert len(set(versions)) == len(versions)

    # The succeeded attempt (2) uploaded its outputs under a0 and referenced them.
    outputs_bytes = _uploaded_bytes(fake_client, dataproxy_service_pb2.ARTIFACT_TYPE_OUTPUTS, ROOT_ACTION_NAME, 2)
    assert outputs_bytes is not None
    outputs = task_common_pb2.Outputs.FromString(outputs_bytes)
    assert {nl.name: nl.value.scalar.primitive.integer for nl in outputs.literals} == {"o0": 7}
    succeeded = next(e for e in attempt2 if e.phase == _PHASE_SUCCEEDED)
    assert succeeded.outputs.output_uri == f"s3://bucket/meta/{ROOT_ACTION_NAME}/2/outputs.pb"


def test_report_html_uploaded_and_referenced(fake_client):
    run = flyte.with_runcontext(mode="local", report=True).run(report_task, x=9)
    assert run.outputs()[0] == 9

    updates = _all_updates(fake_client)
    # The report-generating driver is mapped onto a0; its report uploads under a0.
    assert {u.event.id.name for u in updates} == {ROOT_ACTION_NAME}

    report_bytes = _uploaded_bytes(fake_client, dataproxy_service_pb2.ARTIFACT_TYPE_REPORT, ROOT_ACTION_NAME, 1)
    assert report_bytes is not None
    assert b"report-marker" in report_bytes

    terminal = next(e for e in _events_for(updates, ROOT_ACTION_NAME) if e.phase == _PHASE_SUCCEEDED)
    assert terminal.outputs.report_uri == f"s3://bucket/meta/{ROOT_ACTION_NAME}/1/report.html"
    assert terminal.outputs.output_uri == f"s3://bucket/meta/{ROOT_ACTION_NAME}/1/outputs.pb"


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
# Live report write-through, aborts, and the raw-data-stays-local contract
# ---------------------------------------------------------------------------


def test_live_report_write_through(fake_client):
    """Every mid-run flyte.report.flush() mirrors the current report to the control
    plane; the completion upload remains the final authoritative write."""
    run = flyte.with_runcontext(mode="local", report=True).run(multi_flush_report_task, x=1)
    assert run.outputs()[0] == 1

    report_uploads = [
        (a, att, fake_client.put_payloads[md5])
        for t, a, att, md5 in fake_client.uploads
        if t == dataproxy_service_pb2.ARTIFACT_TYPE_REPORT
    ]
    # At least a mid-run snapshot plus the authoritative completion write.
    assert len(report_uploads) >= 2
    # All report uploads target the driver's (a0's) running attempt.
    assert all((a, att) == (ROOT_ACTION_NAME, 1) for a, att, _ in report_uploads)
    # A mid-run snapshot has only the first flush; the final write has both.
    assert any(b"flush-one" in p and b"flush-two" not in p for _, _, p in report_uploads)
    assert b"flush-one" in report_uploads[-1][2]
    assert b"flush-two" in report_uploads[-1][2]


def test_interrupt_reports_aborted(fake_client):
    """Ctrl+C (cancellation at the runner's await) mid-run reports every in-flight
    action — a0 included — as ABORTED, with a bounded flush, then re-raises."""
    import concurrent.futures

    # The runner re-raises asyncio.CancelledError; the sync syncify bridge surfaces
    # it as concurrent.futures.CancelledError (a distinct class on Python >= 3.14).
    with pytest.raises((asyncio.CancelledError, concurrent.futures.CancelledError)):
        flyte.with_runcontext(mode="local", report=True).run(interrupt_parent, n=1)

    updates = _all_updates(fake_client)
    aborted = {u.event.id.name: u.event for u in updates if u.event.phase == _PHASE_ABORTED}
    # a0 (the mapped driver) and the in-flight child are both aborted.
    assert ROOT_ACTION_NAME in aborted
    assert len(aborted) >= 2
    for event in aborted.values():
        assert "aborted by user (SIGINT)" in event.error_info.message
    # No SUCCEEDED/FAILED terminal was reported for the aborted actions.
    for name in aborted:
        phases = [e.phase for e in _events_for(updates, name)]
        assert _PHASE_SUCCEEDED not in phases
        assert _PHASE_FAILED not in phases


def test_interrupt_reports_aborted_strict(fake_client):
    """Strict mode preserves interrupt semantics: the interrupt propagates (never a
    reporting error) and the aborts are still reported."""
    import concurrent.futures

    with pytest.raises((asyncio.CancelledError, concurrent.futures.CancelledError)):
        flyte.with_runcontext(mode="local", report=True, report_strict=True).run(interrupt_parent, n=1)

    updates = _all_updates(fake_client)
    aborted = {u.event.id.name for u in updates if u.event.phase == _PHASE_ABORTED}
    assert ROOT_ACTION_NAME in aborted


def test_abort_all_skips_terminal_actions():
    reporter = _make_reporter()
    events = []
    reporter._put = events.append

    reporter.record_root_start(task_name="t")
    reporter.record_start(action_id="c1", task_name="child1", parent_id=ROOT_ACTION_NAME)
    reporter.record_complete(action_id="c1")
    reporter.record_start(action_id="c2", task_name="child2", parent_id=ROOT_ACTION_NAME)

    reporter.abort_all(reason="aborted by user (SIGTERM)")

    aborted = [e for e in events if e.phase == _PHASE_ABORTED]
    assert {e.action_name for e in aborted} == {"c2", ROOT_ACTION_NAME}
    # The root's abort is the last transition observed.
    assert aborted[-1].action_name == ROOT_ACTION_NAME
    assert all(e.error == "aborted by user (SIGTERM)" for e in aborted)
    reporter.close(timeout=1)


def test_raw_file_data_never_uploads(fake_client, tmp_path):
    """Locked semantic: only inputs.pb / outputs.pb / report.html ever reach the
    dataproxy. File literals keep their local URIs — raw data stays local."""
    src = tmp_path / "payload.txt"
    src.write_text("raw-bytes-stay-local")

    run = flyte.with_runcontext(mode="local", report=True).run(file_roundtrip, f=File(path=str(src)))
    assert run.outputs()[0].path == str(src)

    # (1) The only UploadMetadata targets ever seen are the three metadata kinds.
    kinds = {t for t, *_ in fake_client.uploads}
    assert kinds <= {
        dataproxy_service_pb2.ARTIFACT_TYPE_INPUTS,
        dataproxy_service_pb2.ARTIFACT_TYPE_OUTPUTS,
        dataproxy_service_pb2.ARTIFACT_TYPE_REPORT,
    }

    # (3) Nothing else was PUT: every signed-PUT payload corresponds to a recorded
    # UploadMetadata request and vice versa.
    assert set(fake_client.put_payloads) == {md5 for *_, md5 in fake_client.uploads}

    # (2) The uploaded outputs.pb still contains the untouched local URI.
    outputs_bytes = _uploaded_bytes(fake_client, dataproxy_service_pb2.ARTIFACT_TYPE_OUTPUTS, ROOT_ACTION_NAME, 1)
    assert outputs_bytes is not None
    outputs = task_common_pb2.Outputs.FromString(outputs_bytes)
    out_uri = outputs.literals[0].value.scalar.blob.uri
    assert str(src) in out_uri
    assert not out_uri.startswith(("s3://", "gs://", "abfs://"))

    # Same for the offloaded inputs.pb.
    inputs_bytes = _uploaded_bytes(fake_client, dataproxy_service_pb2.ARTIFACT_TYPE_INPUTS, ROOT_ACTION_NAME)
    assert inputs_bytes is not None
    inputs = task_common_pb2.Inputs.FromString(inputs_bytes)
    in_uri = inputs.literals[0].value.scalar.blob.uri
    assert str(src) in in_uri
    assert not in_uri.startswith(("s3://", "gs://", "abfs://"))


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


def test_root_terminal_synthesized_only_without_driver():
    """record_root_* synthesis is fallback-only: it emits a0's terminal when no
    driver action ever materialized, and dedupes when a mapped driver delivered it."""
    reporter = _make_reporter()
    events = []
    reporter._put = events.append

    # No driver ever reports (e.g. failure before dispatch): the root failure lands.
    reporter.record_root_start(task_name="t")
    reporter.record_root_failure(error="boom before dispatch")
    assert [e.phase for e in events] == [_PHASE_RUNNING, _PHASE_FAILED]
    assert events[-1].action_name == ROOT_ACTION_NAME
    assert events[-1].error == "boom before dispatch"
    reporter.close(timeout=1)

    # Driver mapped onto a0 and already terminal: root synthesis is a no-op.
    reporter2 = _make_reporter()
    events2 = []
    reporter2._put = events2.append
    reporter2.record_root_start(task_name="t")
    reporter2.record_start(action_id="driver-local-id", task_name="t", task=object())
    reporter2.record_attempt_complete(action_id="driver-local-id", attempt_num=1)
    reporter2.record_complete(action_id="driver-local-id")
    reporter2.record_root_complete()
    assert [(e.action_name, e.phase) for e in events2] == [
        (ROOT_ACTION_NAME, _PHASE_RUNNING),
        (ROOT_ACTION_NAME, _PHASE_RUNNING),
        (ROOT_ACTION_NAME, _PHASE_SUCCEEDED),
    ]
    # Single shared per-attempt version counter across root + mapped driver events.
    assert [e.version for e in events2] == [0, 1, 2]
    reporter2.close(timeout=1)


def test_enqueue_never_raises_after_close():
    reporter = _make_reporter()
    reporter.close(timeout=1)
    # Late events are dropped silently.
    reporter.record_start(action_id="a1", task_name="t")
    reporter.record_complete(action_id="a1")
