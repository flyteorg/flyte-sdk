"""
Unit tests for the "diagnose and recover" parameters on the existing MCP run/action/condition/logs
tools: ``list_runs`` filters, ``get_run`` detail, ``get_run_io`` per action, ``list_actions``
``in_phase``, ``list_conditions`` ``action_name`` + prompt, ``get_logs`` ``show_ts`` and the
``rerun_run`` options.

Each tool is checked three ways: the SDK is called with the right arguments (recorders), bad
arguments are a ``ToolError`` raised before any SDK call, and a call without the new arguments
reaches the SDK exactly as it did before they existed.

Every remote call is monkeypatched; nothing touches the network. Where the SDK object is easy to
build offline (``Run``, ``RunDetails``, ``Action``) the test builds the real one from protobuf
rather than a stand-in, so a tool reaching for an attribute that does not exist fails here.
"""

from __future__ import annotations

import pathlib
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from flyteidl2.common import identifier_pb2, phase_pb2
from flyteidl2.task import run_pb2
from flyteidl2.workflow import run_definition_pb2

import flyte
import flyte.remote
from flyte.ai.mcp import FlyteMCPAppEnvironment
from flyte.ai.mcp._diagnose import MAX_ERROR_CHARS
from flyte.ai.mcp._flyte_mcp_app import ToolError
from flyte.models import ActionPhase
from flyte.remote._common import TimeFilter


def _fn(env: FlyteMCPAppEnvironment, name: str):
    return env._mcp_server._tool_manager._tools[name].fn


@pytest.fixture
def init_config(monkeypatch):
    import flyte._initialize as init_mod

    cfg = init_mod._InitConfig(
        root_dir=pathlib.Path.cwd(),
        org="test-org",
        project="cfg-project",
        domain="cfg-domain",
        client=object(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(init_mod, "_init_config", cfg)
    return cfg


def _env(group: str, **kwargs) -> FlyteMCPAppEnvironment:
    return FlyteMCPAppEnvironment(name="test-mcp", tool_groups=[group], **kwargs)


class _Aio:
    """A syncified SDK callable: ``X.get.aio(...)`` returning a canned value, recording its calls."""

    def __init__(self, value=None):
        self._value = value
        self.calls: list[dict] = []

    async def aio(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self._value() if callable(self._value) else self._value


async def _agen(items):
    for item in items:
        yield item


class _ListAll:
    """A syncified ``listall``: ``.aio(**kwargs)`` is an async iterator over canned items."""

    def __init__(self, items=()):
        self._items = list(items)
        self.calls: list[dict] = []

    def aio(self, **kwargs):
        self.calls.append(kwargs)
        return _agen(self._items)


class _RerunRecorder:
    def __init__(self):
        self.calls: list[tuple] = []

    async def aio(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return SimpleNamespace(name="r2", url="https://example.com/r2")


def _action_pb2(name="a0", run="r1", phase=phase_pb2.ACTION_PHASE_SUCCEEDED, task="my-task", start=None, end=None):
    metadata = run_definition_pb2.ActionMetadata(task=run_definition_pb2.TaskActionMetadata())
    metadata.task.id.name = task
    action = run_definition_pb2.Action(
        id=identifier_pb2.ActionIdentifier(
            run=identifier_pb2.RunIdentifier(org="o", project="p", domain="d", name=run), name=name
        ),
        metadata=metadata,
        status=run_definition_pb2.ActionStatus(phase=phase),
    )
    if start:
        action.status.start_time.FromDatetime(start)
    if end:
        action.status.end_time.FromDatetime(end)
    return action


def _run(name="r1", **kwargs) -> flyte.remote.Run:
    return flyte.remote.Run(pb2=run_definition_pb2.Run(action=_action_pb2(run=name, **kwargs)))


T0 = datetime(2025, 1, 31, 12, 0, 0)
T1 = datetime(2025, 1, 31, 12, 1, 30)
UTC = timezone.utc


# ------------------------------
# list_runs
# ------------------------------


class TestListRuns:
    @pytest.fixture
    def env(self, init_config):
        return _env("run")

    @pytest.fixture
    def listall(self, monkeypatch):
        recorder = _ListAll()
        monkeypatch.setattr(flyte.remote.Run, "listall", recorder)
        return recorder

    @pytest.mark.asyncio
    async def test_defaults_reproduce_the_previous_sdk_call(self, env, listall):
        await _fn(env, "list_runs")()
        assert listall.calls == [
            {
                "task_name": None,
                "sort_by": ("created_at", "desc"),
                "limit": 100,
                "project": "cfg-project",
                "domain": "cfg-domain",
            }
        ]

    @pytest.mark.asyncio
    async def test_filters_reach_the_sdk(self, env, listall):
        await _fn(env, "list_runs")(
            task_name="t",
            limit=7,
            in_phase=["Failed", "timed_out"],
            paused_only=True,
            task_version="v2",
            created_after="2025-01-31T12:00:00Z",
            created_before="2025-02-01",
            updated_after="2025-01-31T12:00:00+02:00",
            updated_before="2025-02-02T00:00:00",
            with_labels={"team": "ml"},
            with_label_keys=["env"],
            project="p",
            domain="d",
        )
        assert listall.calls == [
            {
                "task_name": "t",
                "sort_by": ("created_at", "desc"),
                "limit": 7,
                "project": "p",
                "domain": "d",
                "in_phase": (ActionPhase.FAILED, ActionPhase.TIMED_OUT),
                "paused_actions_only": True,
                "task_version": "v2",
                "created_at": TimeFilter(
                    after=datetime(2025, 1, 31, 12, tzinfo=UTC), before=datetime(2025, 2, 1, tzinfo=UTC)
                ),
                "updated_at": TimeFilter(
                    after=datetime.fromisoformat("2025-01-31T12:00:00+02:00"), before=datetime(2025, 2, 2, tzinfo=UTC)
                ),
                "with_labels": {"team": "ml"},
                "with_label_keys": ["env"],
            }
        ]

    @pytest.mark.asyncio
    async def test_a_single_bound_makes_a_half_open_window(self, env, listall):
        await _fn(env, "list_runs")(created_after="2025-01-31T12:00:00Z")
        assert listall.calls[0]["created_at"] == TimeFilter(after=datetime(2025, 1, 31, 12, tzinfo=UTC))
        assert "updated_at" not in listall.calls[0]

    @pytest.mark.asyncio
    async def test_only_mine_filters_on_the_callers_subject(self, env, listall, monkeypatch):
        user_get = _Aio(SimpleNamespace(subject=lambda: "subject-123"))
        monkeypatch.setattr(flyte.remote.User, "get", user_get)
        await _fn(env, "list_runs")(only_mine=True)
        assert listall.calls[0]["created_by_subject"] == "subject-123"

    @pytest.mark.asyncio
    async def test_only_mine_false_does_not_look_the_user_up(self, env, listall, monkeypatch):
        user_get = _Aio(SimpleNamespace(subject=lambda: "subject-123"))
        monkeypatch.setattr(flyte.remote.User, "get", user_get)
        await _fn(env, "list_runs")()
        assert user_get.calls == []
        assert "created_by_subject" not in listall.calls[0]

    @pytest.mark.asyncio
    async def test_unknown_phase_lists_the_valid_ones(self, env, listall):
        with pytest.raises(ToolError) as exc:
            await _fn(env, "list_runs")(in_phase=["failed", "exploded"])
        message = str(exc.value)
        assert "'exploded'" in message
        assert all(p.value in message for p in ActionPhase)
        assert listall.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("param", ["created_after", "created_before", "updated_after", "updated_before"])
    async def test_bad_timestamp_is_a_tool_error_naming_the_parameter(self, env, listall, param):
        with pytest.raises(ToolError, match=f"{param}='yesterday'.*ISO-8601"):
            await _fn(env, "list_runs")(**{param: "yesterday"})
        assert listall.calls == []

    @pytest.mark.asyncio
    async def test_empty_label_key_is_rejected(self, env, listall):
        with pytest.raises(ToolError, match="with_labels keys must not be empty"):
            await _fn(env, "list_runs")(with_labels={"": "x"})
        assert listall.calls == []

    @pytest.mark.asyncio
    async def test_task_allowlist_does_not_change_listing(self, init_config, monkeypatch):
        # list_runs never consulted the allowlist, and filtering by phase must not start to.
        listall = _ListAll([_run(task="not-on-the-list")])
        monkeypatch.setattr(flyte.remote.Run, "listall", listall)
        env = _env("run", task_allowlist=["d/p/only-this"])
        result = await _fn(env, "list_runs")(in_phase=["failed"])
        assert listall.calls[0]["in_phase"] == (ActionPhase.FAILED,)
        assert [r["task_name"] for r in result] == ["not-on-the-list"]

    @pytest.mark.asyncio
    async def test_entries_keep_the_run_fields_and_add_task_and_times(self, env, monkeypatch):
        run = _run(task="train", start=T0, end=T1)
        monkeypatch.setattr(flyte.remote.Run, "listall", _ListAll([run]))
        [entry] = await _fn(env, "list_runs")()
        assert entry["action"]["id"]["run"]["name"] == "r1"  # the existing to_dict() payload
        assert entry["task_name"] == "train"
        assert entry["started_at"] == "2025-01-31T12:00:00+00:00"
        assert entry["ended_at"] == "2025-01-31T12:01:30+00:00"

    @pytest.mark.asyncio
    async def test_entry_without_to_dict_or_action_keeps_the_minimal_shape(self, env, monkeypatch):
        # The tool tolerates run-like objects that expose only name and url, as it did before.
        monkeypatch.setattr(flyte.remote.Run, "listall", _ListAll([SimpleNamespace(name="r9", url="u9")]))
        assert await _fn(env, "list_runs")() == [{"name": "r9", "url": "u9"}]

    @pytest.mark.asyncio
    async def test_unstarted_run_has_null_times(self, env, monkeypatch):
        monkeypatch.setattr(flyte.remote.Run, "listall", _ListAll([_run(phase=phase_pb2.ACTION_PHASE_QUEUED)]))
        [entry] = await _fn(env, "list_runs")()
        assert entry["started_at"] is None and entry["ended_at"] is None


# ------------------------------
# get_run
# ------------------------------


def _run_details(*, error: run_definition_pb2.ErrorInfo | None = None, labels=None, **kwargs):
    action = _action_pb2(**kwargs)
    details = run_definition_pb2.RunDetails(
        run_spec=run_pb2.RunSpec(labels=run_pb2.Labels(values=labels or {})),
        action=run_definition_pb2.ActionDetails(id=action.id, metadata=action.metadata, status=action.status),
    )
    if error is not None:
        details.action.error_info.CopyFrom(error)
    return flyte.remote.RunDetails(pb2=details)


def _fake_run(details=None, *, phase="FAILED", details_exc: Exception | None = None):
    def _details():
        if details_exc is not None:
            raise details_exc
        return details

    return SimpleNamespace(
        name="r1",
        phase=phase,
        url="https://example.com/r1",
        done=lambda: phase in ("SUCCEEDED", "FAILED"),
        details=_Aio(_details),
    )


class TestGetRun:
    @pytest.fixture
    def env(self, init_config):
        return _env("run")

    @pytest.mark.asyncio
    async def test_failed_run_reports_task_error_times_duration_and_labels(self, env, monkeypatch):
        error = run_definition_pb2.ErrorInfo(message="boom", kind=run_definition_pb2.ErrorInfo.KIND_USER)
        details = _run_details(
            error=error,
            labels={"team": "ml"},
            phase=phase_pb2.ACTION_PHASE_FAILED,
            task="train",
            start=T0,
            end=T1,
        )
        monkeypatch.setattr(flyte.remote.Run, "get", _Aio(_fake_run(details)))

        result = await _fn(env, "get_run")(name="r1")

        assert result == {
            "name": "r1",
            "phase": "FAILED",
            "url": "https://example.com/r1",
            "done": True,
            "task_name": "train",
            "started_at": "2025-01-31T12:00:00+00:00",
            "ended_at": "2025-01-31T12:01:30+00:00",
            "duration_s": 90.0,
            "labels": {"team": "ml"},
            "error": {"kind": "KIND_USER", "message": "boom"},
        }

    @pytest.mark.asyncio
    async def test_error_message_is_truncated(self, env, monkeypatch):
        error = run_definition_pb2.ErrorInfo(message="x" * (MAX_ERROR_CHARS + 500))
        details = _run_details(error=error, phase=phase_pb2.ACTION_PHASE_FAILED, start=T0, end=T1)
        monkeypatch.setattr(flyte.remote.Run, "get", _Aio(_fake_run(details)))

        message = (await _fn(env, "get_run")(name="r1"))["error"]["message"]
        assert message.startswith("x" * MAX_ERROR_CHARS)
        assert len(message) < MAX_ERROR_CHARS + 100
        assert "truncated, 500 more characters" in message

    @pytest.mark.asyncio
    async def test_succeeded_run_has_no_error_key(self, env, monkeypatch):
        details = _run_details(start=T0, end=T1)
        monkeypatch.setattr(flyte.remote.Run, "get", _Aio(_fake_run(details, phase="SUCCEEDED")))
        result = await _fn(env, "get_run")(name="r1")
        assert "error" not in result
        assert result["labels"] == {}

    @pytest.mark.asyncio
    async def test_unstarted_run_has_no_duration(self, env, monkeypatch):
        details = _run_details(phase=phase_pb2.ACTION_PHASE_QUEUED)
        monkeypatch.setattr(flyte.remote.Run, "get", _Aio(_fake_run(details, phase="QUEUED")))
        result = await _fn(env, "get_run")(name="r1")
        assert result["started_at"] is None
        assert result["duration_s"] is None

    @pytest.mark.asyncio
    async def test_detail_failure_keeps_the_status_and_says_why(self, env, monkeypatch):
        monkeypatch.setattr(flyte.remote.Run, "get", _Aio(_fake_run(details_exc=RuntimeError("rpc down"))))
        result = await _fn(env, "get_run")(name="r1")
        assert result["phase"] == "FAILED"
        assert result["details_error"] == "RuntimeError: rpc down"

    @pytest.mark.asyncio
    async def test_makes_one_details_call(self, env, monkeypatch):
        fake = _fake_run(_run_details(start=T0, end=T1))
        getter = _Aio(fake)
        monkeypatch.setattr(flyte.remote.Run, "get", getter)
        await _fn(env, "get_run")(name="r1")
        assert getter.calls == [{"name": "r1"}]
        assert len(fake.details.calls) == 1


# ------------------------------
# get_run_io
# ------------------------------


class _Outputs:
    named_outputs = {"o0": 1}  # noqa: RUF012


class _IO:
    """Shaped like ``Run`` / ``ActionDetails`` for the parts ``get_run_io`` reads."""

    def __init__(self, *, done=True, **names):
        self.done = lambda: done
        self.inputs = _Aio({"x": 1})
        self.outputs = _Aio(_Outputs())
        for key, value in names.items():
            setattr(self, key, value)


class TestGetRunIO:
    @pytest.fixture
    def env(self, init_config):
        return _env("run")

    @pytest.fixture
    def sdk(self, monkeypatch):
        run = _IO(name="r1")
        action = _IO(name="a3", run_name="r1")
        run_get, action_get = _Aio(run), _Aio(action)
        monkeypatch.setattr(flyte.remote.Run, "get", run_get)
        monkeypatch.setattr(flyte.remote.ActionDetails, "get", action_get)
        return SimpleNamespace(run=run, action=action, run_get=run_get, action_get=action_get)

    @pytest.mark.asyncio
    async def test_defaults_read_the_root_run_as_before(self, env, sdk):
        result = await _fn(env, "get_run_io")(name="r1")
        assert result == {"name": "r1", "inputs": {"x": 1}, "outputs": {"o0": 1}}
        assert sdk.run_get.calls == [{"name": "r1"}]
        assert sdk.action_get.calls == []

    @pytest.mark.asyncio
    async def test_action_name_reads_that_actions_data(self, env, sdk):
        result = await _fn(env, "get_run_io")(name="r1", action_name="a3")
        assert result == {"name": "r1", "action_name": "a3", "inputs": {"x": 1}, "outputs": {"o0": 1}}
        assert sdk.action_get.calls == [{"run_name": "r1", "name": "a3"}]
        assert sdk.run_get.calls == []

    @pytest.mark.asyncio
    async def test_unfinished_action_has_null_outputs(self, env, monkeypatch):
        monkeypatch.setattr(flyte.remote.ActionDetails, "get", _Aio(_IO(name="a3", run_name="r1", done=False)))
        result = await _fn(env, "get_run_io")(name="r1", action_name="a3")
        assert result["outputs"] is None
        assert "outputs_error" not in result

    @pytest.mark.asyncio
    async def test_inputs_only_does_not_fetch_outputs(self, env, sdk):
        result = await _fn(env, "get_run_io")(name="r1", inputs_only=True)
        assert result == {"name": "r1", "inputs": {"x": 1}}
        assert sdk.run.outputs.calls == []

    @pytest.mark.asyncio
    async def test_outputs_only_does_not_fetch_inputs(self, env, sdk):
        result = await _fn(env, "get_run_io")(name="r1", outputs_only=True)
        assert result == {"name": "r1", "outputs": {"o0": 1}}
        assert sdk.run.inputs.calls == []

    @pytest.mark.asyncio
    async def test_both_flags_is_a_tool_error_before_any_sdk_call(self, env, sdk):
        with pytest.raises(ToolError, match="cannot both be true"):
            await _fn(env, "get_run_io")(name="r1", inputs_only=True, outputs_only=True)
        assert sdk.run_get.calls == [] and sdk.action_get.calls == []


# ------------------------------
# list_actions
# ------------------------------


class TestListActions:
    @pytest.fixture
    def env(self, init_config):
        return _env("action")

    @pytest.mark.asyncio
    async def test_defaults_reproduce_the_previous_sdk_call(self, env, monkeypatch):
        listall = _ListAll()
        monkeypatch.setattr(flyte.remote.Action, "listall", listall)
        await _fn(env, "list_actions")(run_name="r1")
        assert listall.calls == [{"for_run_name": "r1"}]

    @pytest.mark.asyncio
    async def test_in_phase_is_validated_and_forwarded(self, env, monkeypatch):
        failed = flyte.remote.Action(_action_pb2(name="a2", phase=phase_pb2.ACTION_PHASE_FAILED, start=T0, end=T1))
        listall = _ListAll([failed])
        monkeypatch.setattr(flyte.remote.Action, "listall", listall)

        result = await _fn(env, "list_actions")(run_name="r1", in_phase=["FAILED", "aborted"])

        assert listall.calls == [{"for_run_name": "r1", "in_phase": (ActionPhase.FAILED, ActionPhase.ABORTED)}]
        assert [a["name"] for a in result] == ["a2"]

    @pytest.mark.asyncio
    async def test_unknown_phase_is_a_tool_error(self, env, monkeypatch):
        listall = _ListAll()
        monkeypatch.setattr(flyte.remote.Action, "listall", listall)
        with pytest.raises(ToolError, match=r"Unknown phase 'done'\. Valid phases: .*failed"):
            await _fn(env, "list_actions")(run_name="r1", in_phase=["done"])
        assert listall.calls == []


# ------------------------------
# list_conditions
# ------------------------------


def _condition(prompt="Ship it?", prompt_type=run_definition_pb2.CONDITION_PROMPT_TYPE_MARKDOWN):
    pb2 = _action_pb2(name="a5", phase=phase_pb2.ACTION_PHASE_PAUSED)
    pb2.metadata.condition.name = "approve"
    details = run_definition_pb2.ActionDetails(id=pb2.id, metadata=pb2.metadata, status=pb2.status)
    details.condition.name = "approve"
    details.condition.prompt = prompt
    details.condition.prompt_type = prompt_type
    return flyte.remote.Condition(pb2=pb2), flyte.remote.ActionDetails(pb2=details)


class TestListConditions:
    @pytest.fixture
    def env(self, init_config):
        return _env("condition")

    @pytest.mark.asyncio
    async def test_defaults_reproduce_the_previous_sdk_call(self, env, monkeypatch):
        listall = _ListAll()
        monkeypatch.setattr(flyte.remote.Condition, "listall", listall)
        await _fn(env, "list_conditions")(run_name="r1")
        assert listall.calls == [{"run_name": "r1", "limit": 50}]

    @pytest.mark.asyncio
    async def test_action_name_is_forwarded(self, env, monkeypatch):
        listall = _ListAll()
        monkeypatch.setattr(flyte.remote.Condition, "listall", listall)
        await _fn(env, "list_conditions")(run_name="r1", action_name="a2", limit=5)
        assert listall.calls == [{"run_name": "r1", "limit": 5, "action_name": "a2"}]

    @pytest.mark.asyncio
    async def test_entries_carry_prompt_and_type_and_keep_existing_keys(self, env, monkeypatch):
        cond, details = _condition()
        monkeypatch.setattr(flyte.remote.Condition, "listall", _ListAll([cond]))
        details_get = _Aio(details)
        monkeypatch.setattr(flyte.remote.ActionDetails, "get", details_get)

        [entry] = await _fn(env, "list_conditions")(run_name="r1")

        assert entry["name"] == "approve"
        assert entry["action_name"] == "a5"
        assert entry["run_name"] == "r1"
        assert {"phase", "expected_type"} <= entry.keys()
        assert entry["prompt"] == "Ship it?"
        assert entry["prompt_type"] == "markdown"
        assert "prompt_error" not in entry
        assert details_get.calls == [{"run_name": "r1", "name": "a5"}]

    @pytest.mark.asyncio
    async def test_unset_prompt_type_is_null(self, env, monkeypatch):
        cond, details = _condition(prompt="", prompt_type=run_definition_pb2.CONDITION_PROMPT_TYPE_UNSPECIFIED)
        monkeypatch.setattr(flyte.remote.Condition, "listall", _ListAll([cond]))
        monkeypatch.setattr(flyte.remote.ActionDetails, "get", _Aio(details))
        [entry] = await _fn(env, "list_conditions")(run_name="r1")
        assert entry["prompt"] is None
        assert entry["prompt_type"] is None

    @pytest.mark.asyncio
    async def test_details_without_a_condition_leave_the_prompt_null(self, env, monkeypatch):
        cond, details = _condition()
        details.pb2.ClearField("condition")
        monkeypatch.setattr(flyte.remote.Condition, "listall", _ListAll([cond]))
        monkeypatch.setattr(flyte.remote.ActionDetails, "get", _Aio(details))
        [entry] = await _fn(env, "list_conditions")(run_name="r1")
        assert entry["prompt"] is None
        assert entry["prompt_type"] is None
        assert "prompt_error" not in entry

    @pytest.mark.asyncio
    async def test_prompt_lookup_failure_keeps_the_entry(self, env, monkeypatch):
        cond, _ = _condition()

        def _boom():
            raise RuntimeError("rpc down")

        monkeypatch.setattr(flyte.remote.Condition, "listall", _ListAll([cond]))
        monkeypatch.setattr(flyte.remote.ActionDetails, "get", _Aio(_boom))
        [entry] = await _fn(env, "list_conditions")(run_name="r1")
        assert entry["name"] == "approve"
        assert entry["prompt"] is None
        assert entry["prompt_error"] == "RuntimeError: rpc down"


# ------------------------------
# get_logs
# ------------------------------


class _LogAction:
    def __init__(self):
        self.seen: list[dict] = []
        self.get_logs = SimpleNamespace(aio=self._get_logs)
        self.details = _Aio(SimpleNamespace(logs_available=lambda attempt=None: True, phase="RUNNING"))

    def _get_logs(self, **kwargs):
        self.seen.append(kwargs)
        return _agen(["line"])


class TestGetLogsShowTs:
    @pytest.fixture
    def env(self, init_config):
        return _env("logs")

    @pytest.fixture
    def action(self, monkeypatch):
        action = _LogAction()
        monkeypatch.setattr(flyte.remote.Action, "get", _Aio(action))
        return action

    @pytest.mark.asyncio
    async def test_default_reproduces_the_previous_sdk_call(self, env, action):
        await _fn(env, "get_logs")(run_name="r1", action_name="a1")
        assert action.seen == [{"attempt": None, "filter_system": True}]

    @pytest.mark.asyncio
    async def test_show_ts_is_forwarded(self, env, action):
        await _fn(env, "get_logs")(run_name="r1", action_name="a1", attempt=2, show_ts=True)
        assert action.seen == [{"attempt": 2, "filter_system": True, "show_ts": True}]

    @pytest.mark.asyncio
    async def test_show_ts_on_the_root_action(self, env, monkeypatch):
        action = _LogAction()
        run = SimpleNamespace(action=action, get_logs=action.get_logs)
        monkeypatch.setattr(flyte.remote.Run, "get", _Aio(run))
        await _fn(env, "get_logs")(run_name="r1", show_ts=True)
        assert action.seen == [{"attempt": None, "filter_system": True, "show_ts": True}]


# ------------------------------
# rerun_run
# ------------------------------


class _RunnerFactory:
    """``flyte.with_runcontext``: records its arguments and hands back a runner with a recording ``rerun``."""

    def __init__(self):
        self.calls: list[dict] = []
        self.rerun = _RerunRecorder()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(rerun=self.rerun)


def _run_with_task(task_name):
    return SimpleNamespace(name="r1", action=SimpleNamespace(task_name=task_name))


@pytest.fixture
def rerun_sdk(init_config, monkeypatch):
    rerun, runner = _RerunRecorder(), _RunnerFactory()
    monkeypatch.setattr(flyte, "rerun", rerun)
    monkeypatch.setattr(flyte, "with_runcontext", runner)
    return SimpleNamespace(rerun=rerun, runner=runner)


class TestRerunOptions:
    @pytest.mark.asyncio
    async def test_defaults_reproduce_the_previous_sdk_call(self, rerun_sdk):
        result = await _fn(_env("run"), "rerun_run")(run_name="r1")
        assert result == {"name": "r2", "url": "https://example.com/r2"}
        assert rerun_sdk.rerun.calls == [(("r1",), {})]
        assert rerun_sdk.runner.calls == []

    @pytest.mark.asyncio
    async def test_recover_with_forced_actions_and_missing_outputs(self, rerun_sdk):
        await _fn(_env("run"), "rerun_run")(
            run_name="r1", recover=True, force_rerun_actions=["a3", "a4"], allow_missing_outputs=True
        )
        assert rerun_sdk.rerun.calls == [
            (
                ("r1",),
                {"recover": True, "force_rerun_actions": ["a3", "a4"], "allow_missing_source_outputs": True},
            )
        ]
        assert rerun_sdk.runner.calls == []

    @pytest.mark.asyncio
    async def test_action_name_reruns_that_action(self, rerun_sdk):
        await _fn(_env("run"), "rerun_run")(run_name="r1", action_name="a3")
        assert rerun_sdk.rerun.calls == [(("r1",), {"action_name": "a3"})]

    @pytest.mark.asyncio
    async def test_name_labels_and_env_go_through_the_run_context(self, rerun_sdk):
        await _fn(_env("run"), "rerun_run")(
            run_name="r1", recover=True, name="retry-1", labels={"team": "ml"}, env={"LOG": "debug"}
        )
        assert rerun_sdk.runner.calls == [
            {"mode": "remote", "name": "retry-1", "env_vars": {"LOG": "debug"}, "labels": {"team": "ml"}}
        ]
        assert rerun_sdk.runner.rerun.calls == [(("r1",), {"recover": True})]
        assert rerun_sdk.rerun.calls == []

    @pytest.mark.asyncio
    async def test_only_name_leaves_labels_and_env_unset(self, rerun_sdk):
        await _fn(_env("run"), "rerun_run")(run_name="r1", name="retry-1")
        assert rerun_sdk.runner.calls == [{"mode": "remote", "name": "retry-1", "env_vars": None, "labels": None}]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"force_rerun_actions": ["a3"]}, "force_rerun_actions requires recover"),
            ({"recover": True, "action_name": "a3"}, "action_name cannot be combined with recover"),
            ({"labels": {"": "x"}}, "labels keys must not be empty"),
            ({"env": {"": "x"}}, "env keys must not be empty"),
        ],
    )
    async def test_invalid_combinations_are_tool_errors_before_any_sdk_call(
        self, rerun_sdk, monkeypatch, kwargs, match
    ):
        getter = _Aio(_run_with_task("t"))
        monkeypatch.setattr(flyte.remote.Run, "get", getter)
        with pytest.raises(ToolError, match=match):
            await _fn(_env("run", task_allowlist=["d/p/t"]), "rerun_run")(run_name="r1", **kwargs)
        assert rerun_sdk.rerun.calls == [] and rerun_sdk.runner.calls == [] and getter.calls == []


class TestRerunOptionsAllowlist:
    """The allowlist gate applies to every path through ``rerun_run``, not just the plain one."""

    def _env(self):
        return _env("run", task_allowlist=["d/p/allowed-task"])

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"recover": True},
            {"recover": True, "force_rerun_actions": ["a1"]},
            {"allow_missing_outputs": True},
            {"name": "retry-1"},
            {"labels": {"a": "b"}},
            {"env": {"A": "b"}},
        ],
    )
    async def test_disallowed_root_task_is_rejected_on_every_option(self, rerun_sdk, monkeypatch, kwargs):
        monkeypatch.setattr(flyte.remote.Run, "get", _Aio(_run_with_task("forbidden-task")))
        with pytest.raises(ToolError, match="d/p/forbidden-task is not allowlisted"):
            await _fn(self._env(), "rerun_run")(run_name="r1", project="p", domain="d", **kwargs)
        assert rerun_sdk.rerun.calls == [] and rerun_sdk.runner.calls == []

    @pytest.mark.asyncio
    async def test_allowed_task_reruns_with_options(self, rerun_sdk, monkeypatch):
        monkeypatch.setattr(flyte.remote.Run, "get", _Aio(_run_with_task("allowed-task")))
        await _fn(self._env(), "rerun_run")(run_name="r1", project="p", domain="d", recover=True, name="x")
        assert rerun_sdk.runner.rerun.calls == [(("r1",), {"recover": True})]

    @pytest.mark.asyncio
    async def test_action_name_is_checked_against_that_actions_task(self, rerun_sdk, monkeypatch):
        # The root task is allowed, but the action being re-run is a different, forbidden task.
        run_get = _Aio(_run_with_task("allowed-task"))
        action_get = _Aio(SimpleNamespace(task_name="forbidden-task"))
        monkeypatch.setattr(flyte.remote.Run, "get", run_get)
        monkeypatch.setattr(flyte.remote.Action, "get", action_get)

        with pytest.raises(ToolError, match="forbidden-task is not allowlisted"):
            await _fn(self._env(), "rerun_run")(run_name="r1", project="p", domain="d", action_name="a3")

        assert action_get.calls == [{"run_name": "r1", "name": "a3"}]
        assert run_get.calls == []
        assert rerun_sdk.rerun.calls == []

    @pytest.mark.asyncio
    async def test_allowed_action_task_reruns(self, rerun_sdk, monkeypatch):
        monkeypatch.setattr(flyte.remote.Action, "get", _Aio(SimpleNamespace(task_name="allowed-task")))
        await _fn(self._env(), "rerun_run")(run_name="r1", project="p", domain="d", action_name="a3")
        assert rerun_sdk.rerun.calls == [(("r1",), {"action_name": "a3"})]

    @pytest.mark.asyncio
    async def test_action_with_unknown_task_is_rejected(self, rerun_sdk, monkeypatch):
        monkeypatch.setattr(flyte.remote.Action, "get", _Aio(SimpleNamespace(task_name=None)))
        with pytest.raises(ToolError, match="Cannot determine which task action 'a3' of run 'r1'"):
            await _fn(self._env(), "rerun_run")(run_name="r1", project="p", domain="d", action_name="a3")
        assert rerun_sdk.rerun.calls == []
