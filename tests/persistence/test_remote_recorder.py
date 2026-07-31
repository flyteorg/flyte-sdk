"""Tests for publishing a local run to the control plane.

The recorder talks to ``InternalRunService``, so these tests drive it with a fake client that
records the protos it receives. That is the contract that matters: which RPCs get called, in
what order, and with what identifiers.
"""

import asyncio
import threading

import pytest
from flyteidl2.common import phase_pb2

import flyte._persistence._remote_recorder as rr_mod
from flyte._persistence._recorder import RunRecorder
from flyte._persistence._remote_recorder import PublishStats, RemoteRunRecorder


class _FakeInputs:
    """Stands in for the converted ``Inputs`` wrapper the controller passes through."""

    class _Proto:
        @staticmethod
        def SerializeToString() -> bytes:
            return b"inputs-bytes"

    proto_inputs = _Proto()


class _FakeOutputs:
    """Stands in for the ``Outputs`` wrapper the controller hands to record_complete."""

    class _Proto:
        @staticmethod
        def SerializeToString() -> bytes:
            return b"outputs-bytes"

    proto_outputs = _Proto()


def _patch_upload(monkeypatch, fn):
    """Stub the data proxy's signed-URL upload.

    Patches the plain coroutine rather than the syncified wrapper, matching what the recorder
    calls -- see `_upload_inputs` for why the wrapper is avoided.
    """
    import flyte.remote._data as data_mod

    async def _call(cfg, fp, **kw):
        return await fn(fp, **kw)

    monkeypatch.setattr(data_mod, "_upload_single_file", _call)
    monkeypatch.setattr("flyte._initialize.get_init_config", object)


class _FakeStatus:
    def __init__(self, code=0, message=""):
        self.code = code
        self.message = message


class _FakeResponse:
    def __init__(self, code=0):
        self.status = _FakeStatus(code)


class FakeClient:
    """Captures calls in order. Mirrors the async signature of the generated client."""

    def __init__(self, fail_code: int = 0):
        self.calls: list[tuple[str, object]] = []
        self._fail_code = fail_code
        self._lock = threading.Lock()

    async def record_action(self, request, *, timeout_ms=None, headers=None):
        with self._lock:
            self.calls.append(("record_action", request))
        return _FakeResponse(self._fail_code)

    async def update_action_status(self, request, *, timeout_ms=None, headers=None):
        with self._lock:
            self.calls.append(("update_action_status", request))
        return _FakeResponse(self._fail_code)

    async def record_action_events(self, request, *, timeout_ms=None, headers=None):
        with self._lock:
            self.calls.append(("record_action_events", request))
        return _FakeResponse(self._fail_code)

    def names(self) -> list[str]:
        with self._lock:
            return [n for n, _ in self.calls]

    def of(self, name: str) -> list[object]:
        with self._lock:
            return [r for n, r in self.calls if n == name]


def _recorder(client) -> RemoteRunRecorder:
    return RemoteRunRecorder(
        run_name="r-1",
        project="proj",
        domain="dev",
        org="acme",
        client=client,
    )


class TestRecordAction:
    def test_root_action_creates_run_with_task_spec(self):
        from flyteidl2.task import task_definition_pb2

        client = FakeClient()
        rec = _recorder(client)
        spec = task_definition_pb2.TaskSpec(short_name="my_task")
        rec.record_action_start(action_name="a0", task_name="my_task", version="v1", task_spec=spec)
        stats = rec.close()

        reqs = client.of("record_action")
        assert len(reqs) == 1
        req = reqs[0]
        # Recording a0 is what creates the run -- there is no separate run row.
        assert req.action_id.name == "a0"
        assert req.action_id.run.name == "r-1"
        assert req.action_id.run.project == "proj"
        assert req.action_id.run.domain == "dev"
        assert req.action_id.run.org == "acme"
        assert req.parent == ""
        assert req.task.id.name == "my_task"
        assert req.task.id.version == "v1"
        # The spec carries the code bundle, which is what makes source browsable.
        assert req.task.spec.short_name == "my_task"
        assert stats.ok

    def test_child_action_records_parent_and_no_spec(self):
        client = FakeClient()
        rec = _recorder(client)
        rec.record_action_start(action_name="a1", task_name="child", version="v1", parent="a0", group="g")
        rec.close()

        req = client.of("record_action")[0]
        assert req.action_id.name == "a1"
        assert req.parent == "a0"
        assert req.group == "g"
        assert req.task.id.name == "child"
        # Child actions are recorded by task id only; no local re-serialization.
        assert not req.task.HasField("spec")

    def test_start_marks_action_running(self):
        client = FakeClient()
        rec = _recorder(client)
        rec.record_action_start(action_name="a0", task_name="t", version="v1")
        rec.close()

        # RecordAction must land before the status update for the same action.
        assert client.names() == ["record_action", "update_action_status"]
        status = client.of("update_action_status")[0].status
        assert status.phase == phase_pb2.ActionPhase.ACTION_PHASE_RUNNING


class TestTerminalStates:
    def test_success_uploads_outputs_and_reports_uri(self, monkeypatch):
        """A local run never writes outputs.pb, so publishing must upload it itself."""
        uploaded = []

        async def fake_upload(local, **kwargs):
            uploaded.append((local.name, local.read_bytes()))
            return "md5", "s3://backend-chosen/uploads/outputs.pb"

        _patch_upload(monkeypatch, fake_upload)

        client = FakeClient()
        rec = _recorder(client)
        rec.record_action_start(action_name="a0", task_name="t", version="v1")
        rec.record_action_success(action_name="a0", outputs=_FakeOutputs())
        rec.close()

        phases = [r.status.phase for r in client.of("update_action_status")]
        assert phase_pb2.ActionPhase.ACTION_PHASE_SUCCEEDED in phases

        assert uploaded == [("outputs.pb", b"outputs-bytes")]
        events = client.of("record_action_events")
        assert len(events) == 1
        assert events[0].events[0].outputs.output_uri == "s3://backend-chosen/uploads/outputs.pb"

    def test_success_without_outputs_emits_no_event(self):
        client = FakeClient()
        rec = _recorder(client)
        rec.record_action_start(action_name="a0", task_name="t", version="v1")
        rec.record_action_success(action_name="a0")
        rec.close()

        assert client.of("record_action_events") == []

    def test_failed_output_upload_emits_no_event(self, monkeypatch):
        async def boom(local, **kwargs):
            raise OSError("upload rejected")

        _patch_upload(monkeypatch, boom)
        client = FakeClient()
        rec = _recorder(client)
        rec.record_action_start(action_name="a0", task_name="t", version="v1")
        rec.record_action_success(action_name="a0", outputs=_FakeOutputs())
        stats = rec.close()

        # Better no outputs URI than one the backend cannot read.
        assert client.of("record_action_events") == []
        assert stats.failed >= 1

    def test_failure_sets_phase_and_error_info(self):
        from flyteidl2.workflow import run_definition_pb2

        client = FakeClient()
        rec = _recorder(client)
        rec.record_action_start(action_name="a0", task_name="t", version="v1")
        rec.record_action_failure(action_name="a0", error="boom")
        rec.close()

        phases = [r.status.phase for r in client.of("update_action_status")]
        assert phase_pb2.ActionPhase.ACTION_PHASE_FAILED in phases

        event = client.of("record_action_events")[0].events[0]
        assert event.error_info.message == "boom"
        assert event.error_info.kind == run_definition_pb2.ErrorInfo.Kind.KIND_USER

    def test_system_error_kind(self):
        from flyteidl2.workflow import run_definition_pb2

        client = FakeClient()
        rec = _recorder(client)
        rec.record_action_start(action_name="a0", task_name="t", version="v1")
        rec.record_action_failure(action_name="a0", error="oops", is_system_error=True)
        rec.close()

        event = client.of("record_action_events")[0].events[0]
        assert event.error_info.kind == run_definition_pb2.ErrorInfo.Kind.KIND_SYSTEM

    def test_attempt_count_is_carried_to_terminal_update(self):
        client = FakeClient()
        rec = _recorder(client)
        rec.record_action_start(action_name="a0", task_name="t", version="v1")
        rec.record_attempt(action_name="a0", attempt=3)
        rec.record_action_success(action_name="a0")
        rec.close()

        terminal = [
            r
            for r in client.of("update_action_status")
            if r.status.phase == phase_pb2.ActionPhase.ACTION_PHASE_SUCCEEDED
        ]
        assert terminal[0].status.attempts == 3


class TestInputUpload:
    """Inputs must be uploaded off the RPC path.

    A local run holds inputs in memory, so publishing has to write inputs.pb itself. Doing that
    inline once stalled a whole map task's worth of actions when bucket credentials were
    unusable (~20s per attempt, serialized), so the upload is now concurrent and bounded.
    """

    def test_input_uri_comes_from_signed_url_upload(self, monkeypatch):
        uploaded = []

        async def fake_upload(local, **kwargs):
            uploaded.append(local.name)
            return "md5", "s3://backend-chosen/uploads/inputs.pb"

        _patch_upload(monkeypatch, fake_upload)

        client = FakeClient()
        rec = _recorder(client)
        rec.record_action_start(
            action_name="a1",
            task_name="t",
            version="v1",
            parent="a0",
            inputs=_FakeInputs(),
        )
        rec.close()

        # The URI is whatever the data proxy hands back, not a path we picked.
        assert client.of("record_action")[0].input_uri == "s3://backend-chosen/uploads/inputs.pb"
        assert uploaded == ["inputs.pb"]

    def test_slow_upload_is_bounded_and_action_still_records(self, monkeypatch):
        async def slow_upload(local, **kwargs):
            await asyncio.sleep(60)  # never completes within the upload timeout

        _patch_upload(monkeypatch, slow_upload)
        monkeypatch.setattr(rr_mod, "_UPLOAD_TIMEOUT_SEC", 0.2)

        client = FakeClient()
        rec = _recorder(client)
        for i in range(5):
            rec.record_action_start(
                action_name=f"a{i}",
                task_name="t",
                version="v1",
                parent="a0",
                inputs=_FakeInputs(),
            )
        # Each upload is abandoned at the timeout, so every action still gets recorded.
        rec.close(timeout=20)
        assert len(client.of("record_action")) == 5
        assert all(r.input_uri == "" for r in client.of("record_action"))

    def test_upload_failure_is_counted_not_fatal(self, monkeypatch):
        async def boom(local, **kwargs):
            raise OSError("upload rejected")

        _patch_upload(monkeypatch, boom)

        client = FakeClient()
        rec = _recorder(client)
        rec.record_action_start(action_name="a1", task_name="t", version="v1", inputs=_FakeInputs())
        stats = rec.close()
        # The action itself was still recorded.
        assert len(client.of("record_action")) == 1
        assert stats.failed >= 1

    def test_no_inputs_means_no_input_uri(self):
        client = FakeClient()
        rec = _recorder(client)
        rec.record_action_start(action_name="a1", task_name="t", version="v1")
        rec.close()
        assert client.of("record_action")[0].input_uri == ""


class TestCaptureEnvironment:
    """A local run has no image, so the interpreter snapshot is the only record of what ran."""

    def test_reports_interpreter_and_packages(self):
        from flyte._persistence._remote_recorder import capture_environment

        env = capture_environment()
        assert env["python"].count(".") >= 2
        assert env["executable"]
        assert isinstance(env["in_virtualenv"], bool)
        assert any(n.lower() == "pytest" for n in env["packages"])
        assert env["env_var_names"] == sorted(env["env_var_names"])

    def test_packages_sorted_case_insensitively(self):
        from flyte._persistence._remote_recorder import capture_environment

        names = list(capture_environment()["packages"])
        assert names == sorted(names, key=str.lower)


class TestLocalEnvironmentOnTask:
    """A local run has no image, so the task must describe the interpreter that ran it.

    These assert the spec is actually mutated: the surrounding code degrades quietly when spec
    building fails, which previously hid a bad proto import and shipped specs with nothing on
    them at all.
    """

    @staticmethod
    def _spec():
        from flyteidl2.core import tasks_pb2
        from flyteidl2.task import task_definition_pb2

        return task_definition_pb2.TaskSpec(
            task_template=tasks_pb2.TaskTemplate(container=tasks_pb2.Container(image="repo/img:tag"))
        )

    def test_image_is_cleared(self):
        client = FakeClient()
        rec = _recorder(client)
        spec = self._spec()
        rec._describe_local_environment(spec)
        rec.close()
        # Nothing was containerized; an image would imply a pod that never existed.
        assert spec.task_template.container.image == ""

    def test_interpreter_facts_land_on_the_container(self):
        client = FakeClient()
        rec = _recorder(client)
        spec = self._spec()
        rec._describe_local_environment(spec)
        rec.close()
        env = {e.key: e.value for e in spec.task_template.container.env}
        assert env["FLYTE_LOCAL_RUN"] == "true"
        assert "FLYTE_LOCAL_PYTHON" in env
        assert int(env["FLYTE_LOCAL_PACKAGES"]) > 0

    def test_full_manifest_lands_in_custom(self):
        client = FakeClient()
        rec = _recorder(client)
        spec = self._spec()
        rec._describe_local_environment(spec)
        rec.close()
        custom = spec.task_template.custom
        assert custom["local_run"] is True
        assert len(custom["packages"]) > 0
        assert len(custom["env_var_names"]) > 0

    def test_env_var_values_are_never_published(self, monkeypatch):
        monkeypatch.setenv("FLYTE_TEST_SECRET", "super-secret-value")
        client = FakeClient()
        rec = _recorder(client)
        spec = self._spec()
        rec._describe_local_environment(spec)
        rec.close()
        blob = str(spec)
        # The name is useful context; the value would be a credential leak.
        assert "FLYTE_TEST_SECRET" in blob
        assert "super-secret-value" not in blob


class TestLogLinks:
    """A task's `flyte.Link` entries are the platform's own durable-log mechanism.

    The executor publishes templated links as ActionEvent.log_info and the console renders them
    as links out to whatever system retains the logs; a published local run must carry the
    user's links through rather than drop them.
    """

    def test_links_published_as_task_logs(self):
        client = FakeClient()
        rec = _recorder(client)
        rec.record_action_start(
            action_name="a1",
            task_name="t",
            version="v1",
            parent="a0",
            log_links=[("Grafana", "https://grafana/d/x?pod=p"), ("CloudWatch", "https://console.aws/x")],
        )
        rec.close()

        events = [e.events[0] for e in client.of("record_action_events")]
        links = [ln for e in events for ln in e.log_info]
        assert {(ln.name, ln.uri) for ln in links} == {
            ("Grafana", "https://grafana/d/x?pod=p"),
            ("CloudWatch", "https://console.aws/x"),
        }

    def test_no_links_emits_no_event(self):
        client = FakeClient()
        rec = _recorder(client)
        rec.record_action_start(action_name="a1", task_name="t", version="v1", parent="a0")
        rec.close()
        assert client.of("record_action_events") == []

    def test_supplementary_events_use_distinct_versions(self):
        """Same (action, attempt, phase) with the same version would collide and lose one."""
        client = FakeClient()
        rec = _recorder(client)
        rec.record_action_start(action_name="a0", task_name="t", version="v1", log_links=[("L", "https://x")])
        rec.record_run_artifacts(action_name="a0", report_uri="https://r")
        rec.close()

        events = [e.events[0] for e in client.of("record_action_events")]
        keys = {(e.attempt, e.phase, e.version) for e in events}
        assert len(keys) == len(events), f"supplementary events share a primary key: {keys}"


class TestRunArtifacts:
    def test_report_recorded(self):
        client = FakeClient()
        rec = _recorder(client)
        rec.record_run_artifacts(action_name="a0", report_uri="s3://b/r.html")
        rec.close()
        event = client.of("record_action_events")[0].events[0]
        assert event.outputs.report_uri == "s3://b/r.html"

    def test_nothing_to_attach_sends_nothing(self):
        client = FakeClient()
        rec = _recorder(client)
        rec.record_run_artifacts(action_name="a0")
        rec.close()
        assert client.of("record_action_events") == []


class TestResilience:
    def test_rpc_failure_is_counted_not_raised(self):
        client = FakeClient(fail_code=13)
        rec = _recorder(client)
        rec.record_action_start(action_name="a0", task_name="t", version="v1")
        stats = rec.close()

        assert stats.failed >= 1
        assert not stats.ok
        assert "13" in (stats.first_error or "")

    def test_client_construction_failure_does_not_block_producers(self):
        # No client and no session config: the worker cannot build one.
        rec = RemoteRunRecorder(run_name="r-1", project="p", domain="d")
        for i in range(50):
            rec.record_action_start(action_name=f"a{i}", task_name="t", version="v1")
        stats = rec.close()
        # Producers were never blocked and the failure was reported once.
        assert stats.failed >= 1
        assert stats.sent == 0

    def test_submit_after_close_is_ignored(self):
        client = FakeClient()
        rec = _recorder(client)
        rec.close()
        rec.record_action_start(action_name="a0", task_name="t", version="v1")
        assert client.names() == []

    def test_close_is_idempotent(self):
        client = FakeClient()
        rec = _recorder(client)
        rec.record_action_start(action_name="a0", task_name="t", version="v1")
        first = rec.close()
        second = rec.close()
        assert (first.sent, first.failed) == (second.sent, second.failed)

    def test_exception_from_client_is_captured(self):
        class Boom:
            async def record_action(self, request, *, timeout_ms=None, headers=None):
                raise RuntimeError("network down")

        rec = RemoteRunRecorder(run_name="r", project="p", domain="d", client=Boom())
        rec.record_action_start(action_name="a0", task_name="t", version="v1")
        stats = rec.close()
        assert stats.failed >= 1
        assert "network down" in (stats.first_error or "")


class TestPublishStats:
    @pytest.mark.parametrize(
        "stats,expected",
        [
            (PublishStats(), True),
            (PublishStats(sent=5), True),
            (PublishStats(failed=1), False),
            (PublishStats(dropped=1), False),
        ],
    )
    def test_ok(self, stats, expected):
        assert stats.ok is expected


class TestRunRecorderFanOut:
    """The unified RunRecorder must drive the remote backend alongside tracker/SQLite."""

    def test_is_active_with_remote_only(self):
        client = FakeClient()
        rec = _recorder(client)
        try:
            assert RunRecorder(remote=rec).is_active is True
        finally:
            rec.close()

    def test_root_start_is_local_only(self):
        """Publishing skips the synthetic root: on the backend the run *is* its root task."""
        client = FakeClient()
        remote = _recorder(client)
        r = RunRecorder(remote=remote, version="v9")
        r.record_root_start(task_name="root")
        r.record_root_complete()
        remote.close()

        assert client.names() == []

    def test_first_top_level_action_becomes_the_run_root(self):
        """The root task is published as a0, so the tree matches a remote run's shape."""
        client = FakeClient()
        remote = _recorder(client)
        r = RunRecorder(remote=remote, version="v1")
        r.record_start(action_id="root-action", task_name="main", parent_id=None)
        r.record_start(action_id="child", task_name="fn", parent_id="root-action")
        remote.close()

        reqs = client.of("record_action")
        assert reqs[0].action_id.name == "a0"
        assert reqs[0].parent == ""
        # Children referencing the root by its real name are remapped onto a0.
        assert reqs[1].action_id.name == "child"
        assert reqs[1].parent == "a0"

    def test_failure_forwarded(self):
        client = FakeClient()
        remote = _recorder(client)
        r = RunRecorder(remote=remote)
        r.record_start(action_id="a1", task_name="child", parent_id="a0")
        r.record_failure(action_id="a1", error="bad")
        remote.close()

        event = client.of("record_action_events")[0].events[0]
        assert event.error_info.message == "bad"

    def test_close_remote_returns_none_without_remote(self):
        assert RunRecorder().close_remote() is None
