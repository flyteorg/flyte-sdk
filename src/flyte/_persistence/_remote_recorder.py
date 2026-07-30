"""Publish a locally-executed run to the control plane so it appears like a remote run.

A local run drives itself: nothing in the cluster knows it exists. This recorder reports the
action tree through ``InternalRunService`` — the same RPCs the platform's executor uses — so
``flyte run --local --publish`` shows up in the console with its task tree, inputs/outputs,
code bundle, any configured log links, and a report holding the captured console output.

Design notes:

* A "run" on the backend is just the action named ``a0``; there is no separate run row. So
  recording ``a0`` via ``RecordAction`` *creates* the run, and nothing is ever enqueued for
  execution. We deliberately never call ``RunService.CreateRun``, which would hand the root
  action to the cluster's executor.
* Publishing is best-effort and must never break the user's run. Every RPC failure is counted
  and logged; nothing propagates to task code.
* Recording happens on a single background thread with its own event loop. Actions publish
  concurrently, but work for one action is serialized against a per-action lock, which is what
  guarantees ``RecordAction`` lands before that action's ``UpdateActionStatus``.
* Inputs and outputs are uploaded here, through the data proxy's signed URLs. A local run keeps
  both in memory — only the pod path writes ``inputs.pb``/``outputs.pb`` — so without this the
  console would have nothing to read. Signed URLs also mean publishing needs no local cloud
  credentials, matching how the code bundle is uploaded.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

from flyte._logging import logger

if TYPE_CHECKING:
    from flyte.remote._client.auth._session import SessionConfig

# Bounded so a pathological fan-out (huge map task) can't grow the queue without limit.
# Overflow is dropped and reported rather than blocking the user's run.
_MAX_PENDING = 10_000

# How long ``close()`` waits for the backlog to drain before giving up.
_DEFAULT_CLOSE_TIMEOUT_SEC = 30.0

# Only the first few failures are logged individually; the rest are summarized on close.
_MAX_LOGGED_ERRORS = 3

# Per-RPC ceiling. Without this, one hung call (an endpoint that accepts the connection but never
# answers) consumes the entire close budget and later actions are silently lost.
_RPC_TIMEOUT_MS = 10_000

# Ceiling on a single blob upload, so one slow write cannot stall the rest of the recording.
_UPLOAD_TIMEOUT_SEC = 15.0

# Sentinel distinguishing "queue was empty this poll" from the ``None`` shutdown signal.
_EMPTY = object()

# How often the executor thread wakes to check the queue. See `_next_entry`.
_QUEUE_POLL_SEC = 0.25

# Supplementary events (ones with no phase of their own) are distinguished by `version`, because
# action_events is keyed by (project, domain, run, name, attempt, phase, version) and inserted
# ON CONFLICT DO NOTHING -- two events sharing a key silently lose one while the RPC returns OK.
_VERSION_LOG_LINKS = 1
_VERSION_RUN_ARTIFACTS = 2

# How many actions may be published at once. Each carries up to two signed-URL uploads, so some
# concurrency is what keeps a fan-out inside the close budget; the cap keeps it polite.
_MAX_CONCURRENCY = 8


def capture_environment() -> dict[str, Any]:
    """Snapshot the interpreter and installed packages.

    A remote run records what it ran in via its container image. A published local run has no
    image -- it ran against whatever is installed on the developer's machine -- so without this
    there is nothing to say which versions produced the result.
    """
    import os
    import platform
    import sys

    packages: dict[str, str] = {}
    try:
        from importlib.metadata import distributions

        for dist in distributions():
            try:
                name = dist.metadata["Name"]
            except Exception:
                name = None
            if name:
                # Duplicates are possible across sys.path entries; first wins, matching what an
                # import would actually resolve to.
                packages.setdefault(name, dist.version or "")
    except Exception as e:  # pragma: no cover - metadata backends are not worth failing on
        logger.debug(f"Could not enumerate installed packages: {e}")

    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "in_virtualenv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        "platform": platform.platform(),
        "packages": dict(sorted(packages.items(), key=lambda kv: kv[0].lower())),
        # Names only, never values. Which variables are set is useful for reproducing a run;
        # their contents routinely include credentials, and this is published to anyone who can
        # see the run.
        "env_var_names": sorted(os.environ),
    }


def _serialize_outputs(outputs: Any) -> bytes | None:
    """Serialize an ``Outputs`` wrapper to bytes, or None when there is nothing to upload.

    The backend reads the outputs object as a literal collection, so the wire format matches
    what the pod path writes for a remote run.
    """
    proto = getattr(outputs, "proto_outputs", None)
    if proto is None:
        return None
    try:
        return proto.SerializeToString()
    except Exception:
        return None


@dataclass
class _ActionState:
    """Per-action bookkeeping needed to build later payloads."""

    start_time: float
    attempt: int = 1


@dataclass
class PublishStats:
    """Outcome of a publishing session, for surfacing to the CLI."""

    sent: int = 0
    failed: int = 0
    dropped: int = 0
    first_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.failed == 0 and self.dropped == 0


class RemoteRunRecorder:
    """Records a local run's actions to the control plane. All public methods are sync and
    non-blocking; work is queued to a background thread.
    """

    def __init__(
        self,
        *,
        run_name: str,
        project: str,
        domain: str,
        org: str | None = None,
        session_config: Optional["SessionConfig"] = None,
        client: Any = None,
        serialization_context: Any = None,
    ) -> None:
        self._run_name = run_name
        self._project = project
        self._domain = domain
        self._org = org or ""
        self._session_config = session_config
        self._client = client
        self._s_ctx = serialization_context
        # The first top-level action becomes "a0" -- see `_alias`.
        self._root_alias: str | None = None
        self._spec_cache: dict[str, Any] = {}
        self._env_snapshot: dict[str, Any] | None = None

        self._queue: queue.Queue[Optional[tuple[Callable[[Any], Any], str]]] = queue.Queue(maxsize=_MAX_PENDING)
        self._actions: dict[str, _ActionState] = {}
        self._actions_lock = threading.Lock()
        self._stats = PublishStats()
        self._stats_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._serve,
            name="flyte-run-publisher",
            daemon=True,
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _build_client(self) -> Any:
        """Construct an InternalRunService client bound to this thread's event loop.

        Mirrors ``ControllerClient``: a dedicated connection pool rather than sharing the
        ``ClientSet``'s, since that one belongs to the caller's loop.
        """
        if self._client is not None:
            return self._client
        if self._session_config is None:
            raise RuntimeError("RemoteRunRecorder requires either a session_config or a client")
        from flyteidl2.workflow.internal_run_service_connect import InternalRunServiceClient

        kwargs = dict(self._session_config.connect_kwargs())
        kwargs["http_client"] = self._session_config.new_http_client()
        return InternalRunServiceClient(**kwargs)

    def _serve(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            try:
                client = self._build_client()
            except Exception as e:
                # Without a client nothing can be published. Drain the queue so producers
                # never block, and report once.
                self._note_error(f"failed to create publishing client: {e}")
                self._drain_without_client()
                return
            loop.run_until_complete(self._drain(client))
        finally:
            try:
                loop.close()
            except Exception:
                pass

    async def _drain(self, client: Any) -> None:
        """Process queued work, concurrently across actions but serially within one.

        Each item may upload a blob before its RPC, so draining strictly in order would make a
        run's publishing time the *sum* of every upload -- enough to blow the close budget on a
        modest fan-out. Ordering only actually matters per action (``RecordAction`` must land
        before that action's status updates), so items are keyed by action and serialized
        against a per-action lock while different actions overlap.

        The blocking ``queue.get`` runs in an executor so the loop stays free to service them.
        """
        loop = asyncio.get_running_loop()
        sem = asyncio.Semaphore(_MAX_CONCURRENCY)
        locks: dict[str, asyncio.Lock] = {}
        pending: set[asyncio.Task] = set()

        async def _run(item: Callable[[Any], Any], action_name: str) -> None:
            lock = locks.setdefault(action_name, asyncio.Lock())
            async with sem, lock:
                await self._invoke(item, client)

        while True:
            entry = await loop.run_in_executor(None, self._next_entry)
            if entry is _EMPTY:
                continue
            try:
                if entry is None:
                    break
                item, action_name = entry
                task = asyncio.create_task(_run(item, action_name))
                pending.add(task)
                task.add_done_callback(pending.discard)
            finally:
                self._queue.task_done()

        if pending:
            await asyncio.wait(set(pending))

    def _next_entry(self) -> Any:
        """Poll the queue with a timeout, for the executor thread.

        A plain blocking ``get`` would park a ThreadPoolExecutor thread indefinitely, and those
        threads are *not* daemons -- so a recorder that never reaches ``close()`` (an unexpected
        error between setup and finalize) would keep the interpreter from exiting. Polling keeps
        the thread short-lived and always joinable.
        """
        try:
            return self._queue.get(timeout=_QUEUE_POLL_SEC)
        except queue.Empty:
            return _EMPTY

    async def _invoke(self, item: Callable[[Any], Any], client: Any) -> None:
        try:
            resp = await item(client)
        except Exception as e:
            self._note_error(f"{type(e).__name__}: {e}")
            return
        # These RPCs report failures in the response status rather than raising.
        status = getattr(resp, "status", None)
        code = getattr(status, "code", 0) if status is not None else 0
        if code:
            self._note_error(f"status code={code} message={getattr(status, 'message', '')!r}")
            return
        with self._stats_lock:
            self._stats.sent += 1

    async def _upload_proto(self, payload: bytes, filename: str, action_name: str, what: str) -> str | None:
        """Upload a serialized proto via the data proxy's signed URL; return the URI or None.

        Uses the same signed-URL flow as the code bundle rather than writing with
        ``flyte.storage`` directly, so publishing needs no local cloud credentials.
        """
        import tempfile
        from pathlib import Path

        try:
            # Deliberately the plain coroutine, not the syncified `flyte.remote.upload_file`:
            # that wrapper's `.aio()` hands the work to syncify's shared background loop, which
            # under `flyte run` is the very loop executing the user's task. Queueing behind a
            # running task means every upload hits its timeout instead of taking ~1s.
            from flyte._initialize import get_init_config
            from flyte.remote._data import _upload_single_file

            with tempfile.TemporaryDirectory() as tmp:
                local = Path(tmp) / filename
                local.write_bytes(payload)
                _, uri = await asyncio.wait_for(
                    _upload_single_file(get_init_config(), local), timeout=_UPLOAD_TIMEOUT_SEC
                )
                return uri
        except asyncio.TimeoutError:
            self._note_error(f"timed out uploading {what} for {action_name}")
        except Exception as e:
            # I/O is a display nicety; losing it must not cost us the action record.
            self._note_error(f"failed to upload {what} for {action_name}: {type(e).__name__}: {e}")
        return None

    def _drain_without_client(self) -> None:
        while True:
            item = self._queue.get()
            self._queue.task_done()
            if item is None:
                return

    def _note_error(self, msg: str) -> None:
        with self._stats_lock:
            self._stats.failed += 1
            if self._stats.first_error is None:
                self._stats.first_error = msg
            should_log = self._stats.failed <= _MAX_LOGGED_ERRORS
        if should_log:
            logger.warning(f"Run publishing: {msg}")

    def _submit(self, fn: Callable[[Any], Any], action_name: str) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait((fn, action_name))
        except queue.Full:
            with self._stats_lock:
                self._stats.dropped += 1

    # ------------------------------------------------------------------
    # Identifiers
    # ------------------------------------------------------------------

    def has_action(self, action_name: str) -> bool:
        with self._actions_lock:
            return action_name in self._actions

    def _alias(self, action_name: str) -> str:
        """Map the run's root action onto ``a0``.

        On the backend a run *is* its ``a0`` action. A local run's root task has a generated
        name, so without this the run would show a synthetic ``a0`` with the real root task
        duplicated beneath it -- one level deeper than an equivalent remote run.
        """
        return "a0" if action_name == self._root_alias else action_name

    def _environment(self) -> dict[str, Any]:
        """Interpreter snapshot for this process, captured once."""
        if self._env_snapshot is None:
            self._env_snapshot = capture_environment()
        return self._env_snapshot

    def _describe_local_environment(self, spec: Any) -> None:
        """Replace the container image with a description of where the task actually ran.

        A remote run's task page answers "what did this execute in?" with an image. Nothing was
        containerized here, and ``translate_task_to_wire`` fills in the image the task *would*
        use -- which reads as a pod that never existed. Drop it, and put the interpreter facts in
        its place so the page still answers the question truthfully. The full package manifest
        goes in ``custom``, since hundreds of entries do not belong in ``env``.
        """
        env = self._environment()
        tt = spec.task_template
        try:
            if tt.HasField("container"):
                # KeyValuePair lives in literals_pb2 even though Container does not.
                from flyteidl2.core import literals_pb2

                tt.container.ClearField("image")
                facts = {
                    "FLYTE_LOCAL_RUN": "true",
                    "FLYTE_LOCAL_PYTHON": f"{env['implementation']} {env['python']}",
                    "FLYTE_LOCAL_PLATFORM": env["platform"],
                    "FLYTE_LOCAL_PREFIX": env["prefix"],
                    "FLYTE_LOCAL_PACKAGES": str(len(env.get("packages") or {})),
                }
                for k, v in facts.items():
                    tt.container.env.append(literals_pb2.KeyValuePair(key=k, value=v))
        except Exception as e:  # pragma: no cover - descriptive metadata is never worth failing on
            logger.debug(f"Could not describe local environment on the container: {e}")
        try:
            from google.protobuf import struct_pb2

            payload = struct_pb2.Struct()
            payload.update(
                {
                    "local_run": True,
                    "python": env["python"],
                    "implementation": env["implementation"],
                    "platform": env["platform"],
                    "executable": env["executable"],
                    "prefix": env["prefix"],
                    "in_virtualenv": env["in_virtualenv"],
                    "packages": env.get("packages") or {},
                    # Names only -- values are omitted deliberately, see `capture_environment`.
                    "env_var_names": env.get("env_var_names") or [],
                }
            )
            tt.custom.CopyFrom(payload)
        except Exception as e:  # pragma: no cover - descriptive metadata is never worth failing on
            logger.debug(f"Could not attach local environment to task spec: {e}")

    @staticmethod
    def _with_deck(task_spec: Any) -> Any:
        """Copy *task_spec* with ``generates_deck`` set.

        The console only offers the report view when the task advertises a deck, so without this
        the published report is fetchable over the API but invisible in the UI. Applied to the
        root action alone: capture is run-level, so child actions genuinely have no report and
        advertising one would open an empty view.
        """
        from flyteidl2.task import task_definition_pb2

        copy = task_definition_pb2.TaskSpec()
        copy.CopyFrom(task_spec)
        copy.task_template.metadata.generates_deck.value = True
        return copy

    def _task_spec_for(self, task_template: Any, task_name: str) -> Any:
        """Serialize a task once per name so every action carries its interface.

        The console needs each action's own spec to render its inputs/outputs; a bare task id is
        enough for the tree but leaves the I/O panels empty. Cached because a fan-out repeats the
        same task many times.
        """
        if task_template is None or self._s_ctx is None:
            return None
        if task_name in self._spec_cache:
            return self._spec_cache[task_name]
        try:
            from flyte._internal.runtime.task_serde import translate_task_to_wire

            spec = translate_task_to_wire(task_template, self._s_ctx)
            if spec is not None:
                self._describe_local_environment(spec)
        except Exception as e:
            logger.debug(f"Run publishing: could not serialize spec for {task_name}: {e}")
            spec = None
        self._spec_cache[task_name] = spec
        return spec

    def _action_id(self, action_name: str):
        from flyteidl2.common import identifier_pb2

        return identifier_pb2.ActionIdentifier(
            name=self._alias(action_name),
            run=identifier_pb2.RunIdentifier(
                org=self._org,
                project=self._project,
                domain=self._domain,
                name=self._run_name,
            ),
        )

    def _task_id(self, task_name: str, version: str):
        from flyteidl2.task import task_definition_pb2

        return task_definition_pb2.TaskIdentifier(
            org=self._org,
            project=self._project,
            domain=self._domain,
            name=task_name,
            version=version,
        )

    @staticmethod
    def _ts(epoch_seconds: float | None):
        if epoch_seconds is None:
            return None
        from google.protobuf import timestamp_pb2

        ts = timestamp_pb2.Timestamp()
        ts.FromMilliseconds(int(epoch_seconds * 1000))
        return ts

    # ------------------------------------------------------------------
    # Recording API (sync, non-blocking)
    # ------------------------------------------------------------------

    def record_action_start(
        self,
        *,
        action_name: str,
        task_name: str,
        version: str,
        parent: str | None = None,
        group: str | None = None,
        task_spec: Any = None,
        task_template: Any = None,
        inputs: Any = None,
        log_links: list[tuple[str, str]] | None = None,
        start_time: float | None = None,
    ) -> None:
        """Create the action row. For ``a0`` this also creates the run.

        ``task_spec`` should be supplied for the root action so the console can render the
        task definition and resolve the code bundle. Child actions are recorded by task id
        only, which is enough for the tree, naming and links without re-serializing every
        subtask locally.

        ``inputs`` is the converted ``Inputs`` proto wrapper. A local run keeps inputs in
        memory, so unlike a remote run nothing has written ``inputs.pb`` yet; we upload it
        here (on the worker loop) so the console can render them.
        """
        started = start_time if start_time is not None else time.time()
        # The first action with no parent is the run's root; it is published as "a0".
        if parent is None and self._root_alias is None and action_name != "a0":
            self._root_alias = action_name
        with self._actions_lock:
            self._actions[action_name] = _ActionState(start_time=started)

        if task_spec is None:
            task_spec = self._task_spec_for(task_template, task_name)
        if task_spec is not None and action_name == self._root_alias:
            task_spec = self._with_deck(task_spec)

        if log_links:
            self.record_log_links(action_name=action_name, log_links=log_links)

        async def _call(client: Any):
            from flyteidl2.workflow import internal_run_service_pb2, run_definition_pb2

            input_uri = (
                await self._upload_proto(inputs.proto_inputs.SerializeToString(), "inputs.pb", action_name, "inputs")
                if inputs is not None
                else None
            )

            task_action = run_definition_pb2.TaskAction(id=self._task_id(task_name, version))
            if task_spec is not None:
                task_action.spec.CopyFrom(task_spec)
            req = internal_run_service_pb2.RecordActionRequest(
                action_id=self._action_id(action_name),
                task=task_action,
            )
            if parent:
                req.parent = self._alias(parent)
            elif self._alias(action_name) != "a0":
                # Anything top-level that is not the root still hangs off the run root.
                req.parent = "a0"
            if group:
                req.group = group
            if input_uri:
                req.input_uri = input_uri
            return await client.record_action(req, timeout_ms=_RPC_TIMEOUT_MS)

        self._submit(_call, action_name)
        # A freshly recorded action defaults to an unspecified phase; mark it running so the
        # console shows progress rather than a blank row while the task executes.
        self.record_action_running(action_name=action_name, start_time=started)

    def record_action_running(self, *, action_name: str, attempt: int = 1, start_time: float | None = None) -> None:
        from flyteidl2.common import phase_pb2

        self._update_status(
            action_name=action_name,
            phase=phase_pb2.ActionPhase.ACTION_PHASE_RUNNING,
            attempt=attempt,
            start_time=start_time,
        )

    def record_action_success(
        self,
        *,
        action_name: str,
        attempt: int | None = None,
        outputs: Any = None,
        end_time: float | None = None,
    ) -> None:
        """Mark an action succeeded, uploading its outputs so the console can render them.

        The outputs must be uploaded here: a local run returns them in-process and never writes
        ``outputs.pb`` (that only happens on the pod path, in
        ``taskrunner.extract_download_run_upload``). Reporting a URI derived from the output path
        would point at an object nothing creates, and the backend silently drops an unreadable
        outputs URI, so the console would show no outputs at all.
        """
        from flyteidl2.common import phase_pb2

        state = self._state(action_name)
        resolved_attempt = attempt if attempt is not None else (state.attempt if state else 1)
        finished = end_time if end_time is not None else time.time()

        self._update_status(
            action_name=action_name,
            phase=phase_pb2.ActionPhase.ACTION_PHASE_SUCCEEDED,
            attempt=resolved_attempt,
            end_time=finished,
        )

        payload = _serialize_outputs(outputs)
        if payload is None:
            return

        async def _call(client: Any):
            uri = await self._upload_proto(payload, "outputs.pb", action_name, "outputs")
            if not uri:
                return None
            return await self._send_event(
                client,
                action_name=action_name,
                attempt=resolved_attempt,
                phase=phase_pb2.ActionPhase.ACTION_PHASE_SUCCEEDED,
                end_time=finished,
                output_uri=uri,
            )

        self._submit(_call, action_name)

    def record_action_failure(
        self,
        *,
        action_name: str,
        error: str,
        attempt: int | None = None,
        end_time: float | None = None,
        is_system_error: bool = False,
    ) -> None:
        from flyteidl2.common import phase_pb2

        state = self._state(action_name)
        resolved_attempt = attempt if attempt is not None else (state.attempt if state else 1)
        finished = end_time if end_time is not None else time.time()

        self._update_status(
            action_name=action_name,
            phase=phase_pb2.ActionPhase.ACTION_PHASE_FAILED,
            attempt=resolved_attempt,
            end_time=finished,
        )
        self._record_event(
            action_name=action_name,
            attempt=resolved_attempt,
            phase=phase_pb2.ActionPhase.ACTION_PHASE_FAILED,
            end_time=finished,
            error=error,
            is_system_error=is_system_error,
        )

    def record_attempt(self, *, action_name: str, attempt: int) -> None:
        """Track the current attempt so terminal updates report the right retry count."""
        with self._actions_lock:
            state = self._actions.get(action_name)
            if state is not None:
                state.attempt = attempt

    def record_log_links(self, *, action_name: str, log_links: list[tuple[str, str]], attempt: int = 1) -> None:
        """Publish a task's configured ``flyte.Link`` entries as TaskLogs.

        This is the mechanism the platform itself uses for durable logs: the executor fills URL
        templates and attaches the result as ``ActionEvent.log_info``, and the console renders
        them as links out to whatever system actually retains the output. A published local run
        carries the user's own links through unchanged.
        """

        def _call(client: Any):
            from flyteidl2.core import execution_pb2
            from flyteidl2.workflow import internal_run_service_pb2, run_definition_pb2

            event = run_definition_pb2.ActionEvent(
                id=self._action_id(action_name),
                attempt=attempt,
                version=_VERSION_LOG_LINKS,
                log_info=[execution_pb2.TaskLog(name=name, uri=uri) for name, uri in log_links],
            )
            return client.record_action_events(
                internal_run_service_pb2.RecordActionEventsRequest(events=[event]), timeout_ms=_RPC_TIMEOUT_MS
            )

        self._submit(_call, action_name)

    def record_run_artifacts(
        self,
        *,
        action_name: str,
        report_uri: str | None = None,
        attempt: int = 1,
    ) -> None:
        """Attach the run report to an action.

        Carries an explicit ``version`` because ``action_events`` is keyed by
        ``(project, domain, run, name, attempt, phase, version)`` and inserted with
        ``ON CONFLICT DO NOTHING``: two phase-less events sharing a version collide and the
        second is silently dropped while the RPC still returns OK.
        """
        if not report_uri:
            return

        def _call(client: Any):
            from flyteidl2.task import common_pb2 as task_common_pb2
            from flyteidl2.workflow import internal_run_service_pb2, run_definition_pb2

            event = run_definition_pb2.ActionEvent(
                id=self._action_id(action_name), attempt=attempt, version=_VERSION_RUN_ARTIFACTS
            )
            event.outputs.CopyFrom(task_common_pb2.OutputReferences(report_uri=report_uri))
            return client.record_action_events(
                internal_run_service_pb2.RecordActionEventsRequest(events=[event]), timeout_ms=_RPC_TIMEOUT_MS
            )

        self._submit(_call, action_name)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _state(self, action_name: str) -> _ActionState | None:
        with self._actions_lock:
            return self._actions.get(action_name)

    def _update_status(
        self,
        *,
        action_name: str,
        phase: int,
        attempt: int,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> None:
        state = self._state(action_name)
        started = start_time if start_time is not None else (state.start_time if state else None)

        def _call(client: Any):
            from flyteidl2.workflow import internal_run_service_pb2, run_definition_pb2

            status = run_definition_pb2.ActionStatus(phase=phase, attempts=attempt)
            if (ts := self._ts(started)) is not None:
                status.start_time.CopyFrom(ts)
            if (ts := self._ts(end_time)) is not None:
                status.end_time.CopyFrom(ts)
            return client.update_action_status(
                internal_run_service_pb2.UpdateActionStatusRequest(
                    action_id=self._action_id(action_name),
                    status=status,
                ),
                timeout_ms=_RPC_TIMEOUT_MS,
            )

        self._submit(_call, action_name)

    def _record_event(
        self,
        *,
        action_name: str,
        attempt: int,
        phase: int,
        end_time: float | None = None,
        output_uri: str | None = None,
        error: str | None = None,
        is_system_error: bool = False,
    ) -> None:
        self._submit(
            lambda client: self._send_event(
                client,
                action_name=action_name,
                attempt=attempt,
                phase=phase,
                end_time=end_time,
                output_uri=output_uri,
                error=error,
                is_system_error=is_system_error,
            ),
            action_name,
        )

    async def _send_event(
        self,
        client: Any,
        *,
        action_name: str,
        attempt: int,
        phase: int,
        end_time: float | None = None,
        output_uri: str | None = None,
        error: str | None = None,
        is_system_error: bool = False,
    ):
        """Build and send a single ``ActionEvent``. Runs on the worker loop."""
        from flyteidl2.task import common_pb2 as task_common_pb2
        from flyteidl2.workflow import internal_run_service_pb2, run_definition_pb2

        state = self._state(action_name)
        started = state.start_time if state else None

        event = run_definition_pb2.ActionEvent(id=self._action_id(action_name), attempt=attempt, phase=phase)
        if (ts := self._ts(started)) is not None:
            event.start_time.CopyFrom(ts)
        if (ts := self._ts(end_time)) is not None:
            event.end_time.CopyFrom(ts)
        if output_uri:
            event.outputs.CopyFrom(task_common_pb2.OutputReferences(output_uri=output_uri))
        if error is not None:
            kind = (
                run_definition_pb2.ErrorInfo.Kind.KIND_SYSTEM
                if is_system_error
                else run_definition_pb2.ErrorInfo.Kind.KIND_USER
            )
            event.error_info.CopyFrom(run_definition_pb2.ErrorInfo(message=error, kind=kind))
        return await client.record_action_events(
            internal_run_service_pb2.RecordActionEventsRequest(events=[event]), timeout_ms=_RPC_TIMEOUT_MS
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def close(self, timeout: float = _DEFAULT_CLOSE_TIMEOUT_SEC) -> PublishStats:
        """Flush the backlog and stop the worker. Returns the publishing outcome."""
        if self._closed:
            return self.stats
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # Queue is saturated; the worker will still exit when it drains to the sentinel
            # we cannot enqueue, so fall back to a blocking put bounded by the timeout.
            try:
                self._queue.put(None, timeout=timeout)
            except queue.Full:
                pass
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning(f"Run publishing did not finish within {timeout:.0f}s; some actions may be missing.")
        stats = self.stats
        if stats.dropped:
            logger.warning(f"Run publishing dropped {stats.dropped} update(s) due to backpressure.")
        if stats.failed > _MAX_LOGGED_ERRORS:
            logger.warning(f"Run publishing had {stats.failed} failures; first was: {stats.first_error}")
        return stats

    @property
    def stats(self) -> PublishStats:
        with self._stats_lock:
            return PublishStats(
                sent=self._stats.sent,
                failed=self._stats.failed,
                dropped=self._stats.dropped,
                first_error=self._stats.first_error,
            )
