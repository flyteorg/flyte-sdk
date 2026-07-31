"""Report local-run state to the control plane via ``LocalRunService``.

``RemoteRunReporter`` is a third :class:`~flyte._persistence._recorder.RunRecorder`
backend (next to the TUI tracker and the SQLite ``RunStore``). Recorder methods are
synchronous and are invoked from async controller code as well as background task
threads, so every ``record_*`` call here only captures a lightweight, fully-computed
event and enqueues it on a thread-safe queue. A dedicated background worker thread
(with its own event loop, mirroring the remote controller's worker model) batches the
queued events into ``LocalRunService.ReportActions`` calls and performs the
outputs/report signed-URL uploads for terminal events.

Reporting is strictly best-effort: enqueue methods never raise, the worker retries a
bounded number of times and then drops the batch with a warning, and the terminal
flush (:meth:`RemoteRunReporter.close`) waits a bounded amount of time so
``flyte run --local`` never hangs on a slow control plane.
"""

from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from flyte._logging import logger

if TYPE_CHECKING:
    from flyteidl2.common import identifier_pb2
    from flyteidl2.workflow import local_run_service_pb2

    from flyte._task import TaskTemplate
    from flyte.remote._client.controlplane import ClientSet

ROOT_ACTION_NAME = "a0"
_MAX_RUN_NAME_LENGTH = 30
# Run names starting with these prefixes are reserved by the platform.
_RESERVED_RUN_NAME_PREFIXES = ("u", "r")

# flyteidl2.common.ActionPhase values (kept as plain ints so this module stays cheap to import).
_PHASE_RUNNING = 4
_PHASE_SUCCEEDED = 5
_PHASE_FAILED = 6
_TERMINAL_PHASES = (_PHASE_SUCCEEDED, _PHASE_FAILED)

_SEND_MAX_RETRIES = 3
_SEND_BACKOFF_SEC = 0.5
_DEFAULT_FLUSH_TIMEOUT_SEC = 30.0

# Version used for task identifiers of locally executed tasks (mirrors the local
# TaskContext version).
_LOCAL_TASK_VERSION = "na"


def generate_local_run_name() -> str:
    """Generate a run name that satisfies the local-run naming contract.

    The control plane requires run names of at most 30 characters that do not start
    with a reserved prefix ('u' or 'r').
    """
    return f"local-{uuid.uuid4().hex[:8]}"


def validate_local_run_name(name: str) -> None:
    """Validate a user-supplied run name against the local-run naming contract.

    :raises ValueError: when the name is too long or starts with a reserved prefix.
    """
    if len(name) > _MAX_RUN_NAME_LENGTH:
        raise ValueError(
            f"Run name {name!r} is too long for local-run reporting ({len(name)} > {_MAX_RUN_NAME_LENGTH} characters)."
        )
    if name.startswith(_RESERVED_RUN_NAME_PREFIXES):
        raise ValueError(
            f"Run name {name!r} is invalid for local-run reporting: names starting with "
            f"{' or '.join(repr(p) for p in _RESERVED_RUN_NAME_PREFIXES)} are reserved by the platform."
        )


class _Stop:
    """Sentinel enqueued by ``close`` to stop the worker after a final drain."""


_STOP = _Stop()


@dataclass
class _Event:
    """A lightweight, fully-computed record of one recorder callback."""

    action_name: str
    phase: int
    attempt: int
    version: int
    timestamp: datetime
    error: str | None = None
    # Serialized ``flyteidl2.task.Outputs`` bytes; serialized at enqueue time so the
    # worker never touches protos shared with the caller thread.
    outputs_bytes: bytes | None = None
    # First-report-only metadata.
    first_report: bool = False
    parent_name: str = ""
    group: str = ""
    task_name: str = ""
    # Terminal-artifact metadata.
    output_path: str | None = None
    has_report: bool = False
    start_time: datetime | None = None


@dataclass
class _ActionInfo:
    """Per-action reporting state, mutated only under the reporter lock."""

    task_name: str
    parent_name: str
    group: str
    start_time: datetime
    output_path: str | None = None
    has_report: bool = False
    attempt: int = 1
    last_phase: int | None = None
    started: bool = False
    # Monotonic event-version counter per attempt.
    versions: dict[int, int] = field(default_factory=dict)


class RemoteRunReporter:
    """Fire-and-forget sink that mirrors the ``RunRecorder`` surface onto ``ReportActions``."""

    def __init__(
        self,
        client: ClientSet,
        run_id: identifier_pb2.RunIdentifier,
        *,
        flush_timeout_sec: float = _DEFAULT_FLUSH_TIMEOUT_SEC,
        verify_ssl: bool = True,
    ) -> None:
        self._client = client
        self._run_id = run_id
        self._flush_timeout = flush_timeout_sec
        self._verify_ssl = verify_ssl
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._actions: dict[str, _ActionInfo] = {}
        self._closed = threading.Event()
        self._done = threading.Event()
        # Artifacts (outputs/report) already uploaded, keyed by (action, attempt, kind).
        self._uploaded: set[tuple[str, int, str]] = set()
        self._worker = threading.Thread(
            target=self._worker_main,
            daemon=True,
            name=f"flyte-local-run-reporter-{run_id.name}",
        )
        self._worker.start()

    # ------------------------------------------------------------------
    # RunRecorder-facing surface (synchronous, never raises, never blocks)
    # ------------------------------------------------------------------

    def get_action(self, action_id: str) -> Any:
        """Return a truthy marker when the action was already reported (used by the
        recorder for parent-chain detection when no TUI tracker is attached)."""
        with self._lock:
            return self._actions.get(action_id)

    def record_start(
        self,
        *,
        action_id: str,
        task_name: str,
        parent_id: str | None = None,
        output_path: str | None = None,
        has_report: bool = False,
        group: str | None = None,
        **_: Any,
    ) -> None:
        try:
            now = datetime.now(timezone.utc)
            with self._lock:
                info = self._actions.get(action_id)
                first = info is None
                if info is None:
                    # The root has no parent; every other action defaults to nesting under it.
                    parent = "" if action_id == ROOT_ACTION_NAME else (parent_id or ROOT_ACTION_NAME)
                    info = _ActionInfo(
                        task_name=task_name,
                        parent_name=parent,
                        group=group or "",
                        start_time=now,
                        output_path=output_path,
                        has_report=has_report,
                        started=True,
                    )
                    self._actions[action_id] = info
                ev = self._make_event(action_id, info, _PHASE_RUNNING, now, first_report=first)
            self._put(ev)
        except Exception as e:
            logger.debug(f"Local-run reporter failed to record start for {action_id}: {e}")

    def record_complete(self, *, action_id: str, outputs: Any = None) -> None:
        self._record_terminal(action_id, _PHASE_SUCCEEDED, outputs=outputs)

    def record_failure(self, *, action_id: str, error: str) -> None:
        self._record_terminal(action_id, _PHASE_FAILED, error=error)

    def record_attempt_start(self, *, action_id: str, attempt_num: int) -> None:
        try:
            now = datetime.now(timezone.utc)
            with self._lock:
                info = self._actions.get(action_id)
                if info is None:
                    return
                info.attempt = max(attempt_num, 1)
                if attempt_num <= 1:
                    # record_start already reported RUNNING for attempt 1.
                    return
                ev = self._make_event(action_id, info, _PHASE_RUNNING, now)
            self._put(ev)
        except Exception as e:
            logger.debug(f"Local-run reporter failed to record attempt start for {action_id}: {e}")

    def record_attempt_complete(self, *, action_id: str, attempt_num: int, outputs: Any = None) -> None:
        self._record_terminal(action_id, _PHASE_SUCCEEDED, outputs=outputs, attempt_num=attempt_num)

    def record_attempt_failure(self, *, action_id: str, attempt_num: int, error: str) -> None:
        # An attempt failure is not necessarily action-terminal (a retry may follow);
        # the subsequent record_attempt_start reports the next attempt as RUNNING,
        # matching platform semantics. A final record_failure for the same attempt is
        # deduplicated by the last_phase check in _record_terminal.
        self._record_terminal(action_id, _PHASE_FAILED, error=error, attempt_num=attempt_num, terminal=False)

    # -- Root ("a0") action ------------------------------------------------

    def record_root_start(self, *, task_name: str) -> None:
        self.record_start(action_id=ROOT_ACTION_NAME, task_name=task_name, parent_id="")

    def record_root_complete(self) -> None:
        self._record_terminal(ROOT_ACTION_NAME, _PHASE_SUCCEEDED)

    def record_root_failure(self, *, error: str) -> None:
        self._record_terminal(ROOT_ACTION_NAME, _PHASE_FAILED, error=error)

    # ------------------------------------------------------------------
    # Flush / shutdown
    # ------------------------------------------------------------------

    def close(self, timeout: float | None = None) -> None:
        """Flush all pending events and stop the worker. Bounded wait; never raises."""
        try:
            if self._closed.is_set():
                self._done.wait(timeout if timeout is not None else self._flush_timeout)
                return
            self._closed.set()
            self._queue.put(_STOP)
            if not self._done.wait(timeout if timeout is not None else self._flush_timeout):
                logger.warning(
                    "Timed out flushing local-run reports to the control plane; "
                    "the reported run state may be incomplete."
                )
        except Exception as e:
            logger.warning(f"Failed to flush local-run reports: {e}")

    async def aclose(self, timeout: float | None = None) -> None:
        """Async wrapper around :meth:`close` so callers on an event loop don't block it."""
        import asyncio

        try:
            await asyncio.to_thread(self.close, timeout)
        except Exception as e:
            logger.warning(f"Failed to flush local-run reports: {e}")

    # ------------------------------------------------------------------
    # Internals — enqueue side
    # ------------------------------------------------------------------

    def _record_terminal(
        self,
        action_id: str,
        phase: int,
        *,
        outputs: Any = None,
        error: str | None = None,
        attempt_num: int | None = None,
        terminal: bool = True,
    ) -> None:
        try:
            now = datetime.now(timezone.utc)
            outputs_bytes: bytes | None = None
            proto_outputs = getattr(outputs, "proto_outputs", None)
            if proto_outputs is not None:
                outputs_bytes = proto_outputs.SerializeToString()
            with self._lock:
                info = self._actions.get(action_id)
                if info is None:
                    # Terminal report for an action we never saw start; register it so
                    # the event still references a known parent chain.
                    info = _ActionInfo(
                        task_name=action_id,
                        parent_name="" if action_id == ROOT_ACTION_NAME else ROOT_ACTION_NAME,
                        group="",
                        start_time=now,
                    )
                    self._actions[action_id] = info
                if attempt_num is not None:
                    info.attempt = max(attempt_num, 1)
                if terminal and info.last_phase == phase:
                    # e.g. record_complete right after record_attempt_complete — already reported.
                    return
                ev = self._make_event(
                    action_id,
                    info,
                    phase,
                    now,
                    error=error,
                    outputs_bytes=outputs_bytes,
                )
            self._put(ev)
        except Exception as e:
            logger.debug(f"Local-run reporter failed to record terminal event for {action_id}: {e}")

    def _make_event(
        self,
        action_id: str,
        info: _ActionInfo,
        phase: int,
        now: datetime,
        *,
        error: str | None = None,
        outputs_bytes: bytes | None = None,
        first_report: bool = False,
    ) -> _Event:
        """Build the event under the caller's lock, advancing the per-attempt version."""
        version = info.versions.get(info.attempt, 0)
        info.versions[info.attempt] = version + 1
        info.last_phase = phase
        return _Event(
            action_name=action_id,
            phase=phase,
            attempt=info.attempt,
            version=version,
            timestamp=now,
            error=error,
            outputs_bytes=outputs_bytes,
            first_report=first_report,
            parent_name=info.parent_name,
            group=info.group,
            task_name=info.task_name,
            output_path=info.output_path,
            has_report=info.has_report,
            start_time=info.start_time,
        )

    def _put(self, ev: _Event) -> None:
        if self._closed.is_set():
            logger.debug(f"Local-run reporter already closed; dropping event for {ev.action_name}")
            return
        self._queue.put(ev)

    # ------------------------------------------------------------------
    # Internals — worker side
    # ------------------------------------------------------------------

    def _worker_main(self) -> None:
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            stop = False
            while not stop:
                item = self._queue.get()
                batch: list[_Event] = []
                if isinstance(item, _Stop):
                    stop = True
                else:
                    batch.append(item)
                # Batch whatever else has already been queued.
                while True:
                    try:
                        nxt = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if isinstance(nxt, _Stop):
                        stop = True
                    else:
                        batch.append(nxt)
                if batch:
                    try:
                        loop.run_until_complete(self._process_batch(batch))
                    except Exception as e:
                        logger.warning(f"Failed to report {len(batch)} local-run event(s): {e}")
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Local-run reporter worker stopped unexpectedly: {e}")
        finally:
            loop.close()
            self._done.set()

    async def _process_batch(self, events: list[_Event]) -> None:
        from flyteidl2.workflow import local_run_service_pb2

        updates = []
        for ev in events:
            try:
                updates.append(await self._build_update(ev))
            except Exception as e:
                logger.warning(f"Skipping local-run report for action {ev.action_name}: {e}")
        if not updates:
            return
        req = local_run_service_pb2.ReportLocalActionsRequest(run_id=self._run_id, updates=updates)
        await self._send_with_retries(req)

    async def _send_with_retries(self, req: local_run_service_pb2.ReportLocalActionsRequest) -> None:
        import asyncio

        last_err: Exception | None = None
        for attempt in range(_SEND_MAX_RETRIES):
            try:
                resp = await self._client.local_run_service.report_actions(req)
                for i, status in enumerate(resp.statuses):
                    if status.code != 0:
                        action = req.updates[i].event.id.name if i < len(req.updates) else "?"
                        logger.warning(
                            f"Control plane rejected local-run report for action {action!r}: "
                            f"{status.message} (code={status.code})"
                        )
                return
            except Exception as e:
                last_err = e
                if attempt < _SEND_MAX_RETRIES - 1:
                    await asyncio.sleep(_SEND_BACKOFF_SEC * (2**attempt))
        logger.warning(
            f"Dropping {len(req.updates)} local-run report(s) after {_SEND_MAX_RETRIES} attempts: {last_err}"
        )

    async def _build_update(self, ev: _Event) -> local_run_service_pb2.LocalActionUpdate:
        from flyteidl2.common import identifier_pb2
        from flyteidl2.task import common_pb2 as task_common_pb2
        from flyteidl2.workflow import local_run_service_pb2, run_definition_pb2

        event = run_definition_pb2.ActionEvent(
            id=identifier_pb2.ActionIdentifier(run=self._run_id, name=ev.action_name),
            attempt=ev.attempt,
            phase=ev.phase,  # type: ignore[arg-type]
            version=ev.version,
        )
        event.reported_time.FromDatetime(ev.timestamp)
        event.updated_time.FromDatetime(ev.timestamp)
        if ev.error:
            event.error_info.message = ev.error
            event.error_info.kind = run_definition_pb2.ErrorInfo.KIND_USER

        output_uri = ""
        report_uri = ""
        if ev.phase in _TERMINAL_PHASES:
            output_uri, report_uri = await self._upload_terminal_artifacts(ev)
        if output_uri or report_uri:
            event.outputs.CopyFrom(task_common_pb2.OutputReferences(output_uri=output_uri, report_uri=report_uri))

        update = local_run_service_pb2.LocalActionUpdate(event=event)
        status = run_definition_pb2.ActionStatus(phase=ev.phase, attempts=ev.attempt)  # type: ignore[arg-type]
        if ev.start_time is not None:
            status.start_time.FromDatetime(ev.start_time)
        if ev.phase in _TERMINAL_PHASES:
            status.end_time.FromDatetime(ev.timestamp)
        update.status.CopyFrom(status)
        if ev.first_report:
            update.parent_name = ev.parent_name
            update.group = ev.group
            if ev.action_name != ROOT_ACTION_NAME:
                update.task.CopyFrom(self._minimal_task_action(ev.task_name))
        return update

    async def _upload_terminal_artifacts(self, ev: _Event) -> tuple[str, str]:
        """Upload outputs.pb / report.html for a terminal event, returning their native URLs.

        Upload failures never fail reporting — the event is still sent, just without
        the corresponding artifact reference.
        """
        from flyteidl2.common import identifier_pb2
        from flyteidl2.dataproxy import dataproxy_service_pb2

        from flyte._persistence._remote_upload import upload_metadata_artifact

        attempt_id = identifier_pb2.ActionAttemptIdentifier(
            action_id=identifier_pb2.ActionIdentifier(run=self._run_id, name=ev.action_name),
            attempt=ev.attempt,
        )
        output_uri = ""
        report_uri = ""

        if ev.outputs_bytes is not None and (ev.action_name, ev.attempt, "outputs") not in self._uploaded:
            try:
                output_uri = await upload_metadata_artifact(
                    self._client.dataproxy_service,
                    artifact_type=int(dataproxy_service_pb2.ARTIFACT_TYPE_OUTPUTS),
                    data=ev.outputs_bytes,
                    action_attempt_id=attempt_id,
                    verify=self._verify_ssl,
                )
                self._uploaded.add((ev.action_name, ev.attempt, "outputs"))
            except Exception as e:
                logger.warning(f"Failed to upload outputs for local-run action {ev.action_name}: {e}")

        if ev.has_report and ev.output_path and (ev.action_name, ev.attempt, "report") not in self._uploaded:
            try:
                report_bytes = self._read_report(ev.output_path)
                if report_bytes:
                    report_uri = await upload_metadata_artifact(
                        self._client.dataproxy_service,
                        artifact_type=int(dataproxy_service_pb2.ARTIFACT_TYPE_REPORT),
                        data=report_bytes,
                        action_attempt_id=attempt_id,
                        verify=self._verify_ssl,
                    )
                    self._uploaded.add((ev.action_name, ev.attempt, "report"))
            except Exception as e:
                logger.warning(f"Failed to upload report for local-run action {ev.action_name}: {e}")

        return output_uri, report_uri

    @staticmethod
    def _read_report(output_path: str) -> bytes | None:
        import pathlib

        from flyte._internal.runtime.io import _REPORT_FILE_NAME
        from flyte.storage._storage import strip_file_header

        report_file = pathlib.Path(strip_file_header(output_path)) / _REPORT_FILE_NAME
        if report_file.is_file():
            return report_file.read_bytes()
        return None

    def _minimal_task_action(self, task_name: str):
        """A minimal TaskAction spec so the console can name the action.

        The full local task template is not serializable without a deploy-time
        serialization context, so only the identifier (and type) is carried.
        """
        from flyteidl2.core import identifier_pb2 as core_identifier_pb2
        from flyteidl2.core import tasks_pb2
        from flyteidl2.task import task_definition_pb2
        from flyteidl2.workflow import run_definition_pb2

        template = tasks_pb2.TaskTemplate(
            id=core_identifier_pb2.Identifier(
                resource_type=core_identifier_pb2.ResourceType.TASK,
                org=self._run_id.org,
                project=self._run_id.project,
                domain=self._run_id.domain,
                name=task_name,
                version=_LOCAL_TASK_VERSION,
            ),
            type="python-task",
        )
        return run_definition_pb2.TaskAction(spec=task_definition_pb2.TaskSpec(task_template=template))


async def start_local_run_reporting(
    *,
    client: ClientSet,
    task: TaskTemplate,
    run_name: str,
    org: str | None,
    project: str,
    domain: str,
    run_spec: Any,
    labels: dict[str, str] | None,
    run_start_time: datetime,
    args: tuple,
    kwargs: dict,
    root_dir: Any = None,
    verify_ssl: bool = True,
) -> RemoteRunReporter | None:
    """Register a local run with the control plane and return its reporter sink.

    Uploads the root inputs (when present), calls ``LocalRunService.CreateRun`` and,
    on success, constructs the :class:`RemoteRunReporter` that streams subsequent
    action state. Returns ``None`` (with a warning) on any failure — reporting must
    never fail or block the local run.
    """
    from flyteidl2.common import identifier_pb2
    from flyteidl2.common import run_pb2 as common_run_pb2
    from flyteidl2.dataproxy import dataproxy_service_pb2
    from flyteidl2.workflow import local_run_service_pb2

    from flyte._internal.runtime import convert
    from flyte._persistence._remote_upload import upload_metadata_artifact

    run_id = identifier_pb2.RunIdentifier(org=org or "", project=project, domain=domain, name=run_name)

    try:
        task_spec = _build_root_task_spec(task, org=org, project=project, domain=domain, root_dir=root_dir)

        offloaded: common_run_pb2.OffloadedInputData | None = None
        inputs = await convert.convert_from_native_to_inputs(task.native_interface, *args, **kwargs)
        if inputs.proto_inputs.literals:
            inputs_bytes = inputs.proto_inputs.SerializeToString()
            inputs_hash = convert.generate_inputs_hash_from_proto(inputs.proto_inputs)
            if not inputs_hash:
                import hashlib

                inputs_hash = hashlib.md5(inputs_bytes).hexdigest()
            native_url = await upload_metadata_artifact(
                client.dataproxy_service,
                artifact_type=int(dataproxy_service_pb2.ARTIFACT_TYPE_INPUTS),
                data=inputs_bytes,
                action_id=identifier_pb2.ActionIdentifier(run=run_id, name=ROOT_ACTION_NAME),
                verify=verify_ssl,
            )
            offloaded = common_run_pb2.OffloadedInputData(uri=native_url, inputs_hash=inputs_hash)

        req = local_run_service_pb2.CreateLocalRunRequest(
            run_id=run_id,
            task_spec=task_spec,
            offloaded_input_data=offloaded,
            run_spec=run_spec,
            labels=labels or {},
        )
        req.run_start_time.FromDatetime(run_start_time)
        await client.local_run_service.create_run(req)
    except Exception as e:
        logger.warning(f"Failed to register local run {run_name!r} with the control plane; running unreported: {e}")
        return None

    return RemoteRunReporter(client, run_id, verify_ssl=verify_ssl)


def _build_root_task_spec(
    task: TaskTemplate, *, org: str | None, project: str, domain: str, root_dir: Any = None
) -> Any:
    """Build the TaskSpec sent on CreateRun for the root task.

    Prefers a full ``translate_task_to_wire`` serialization (no code bundle / image
    cache — image references resolve to their locally computed URIs). Some local-only
    task shapes cannot be serialized that way, so this falls back to a minimal spec
    carrying the identifier and typed interface.
    """
    from flyte.models import SerializationContext

    s_ctx = SerializationContext(
        version=_LOCAL_TASK_VERSION,
        org=org,
        project=project,
        domain=domain,
        root_dir=root_dir,
    )
    try:
        from flyte._internal.runtime.task_serde import translate_task_to_wire

        return translate_task_to_wire(task, s_ctx)
    except Exception as e:
        logger.debug(f"Falling back to a minimal task spec for local-run reporting: {e}")

    from flyteidl2.core import identifier_pb2 as core_identifier_pb2
    from flyteidl2.core import tasks_pb2
    from flyteidl2.task import task_definition_pb2

    template = tasks_pb2.TaskTemplate(
        id=core_identifier_pb2.Identifier(
            resource_type=core_identifier_pb2.ResourceType.TASK,
            org=org or "",
            project=project,
            domain=domain,
            name=task.name,
            version=_LOCAL_TASK_VERSION,
        ),
        type=getattr(task, "task_type", "python-task") or "python-task",
    )
    try:
        from flyte._internal.runtime.types_serde import transform_native_to_typed_interface

        iface = transform_native_to_typed_interface(task.interface)
        if iface is not None:
            template.interface.CopyFrom(iface)
    except Exception:
        pass
    return task_definition_pb2.TaskSpec(task_template=template, short_name=task.short_name[:63])
