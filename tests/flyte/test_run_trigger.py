"""Unit tests for firing a deployed trigger on demand via flyte.run(trigger, ...).

The run is created *as* the trigger (CreateRunRequest.trigger_name) with the trigger's registered
inputs and RunSpec as the floor; keyword inputs override individual inputs and with_runcontext
overrides layer on top of the run spec.
"""

import mock
import pytest
from flyteidl2.common import identifier_pb2
from flyteidl2.common import run_pb2 as common_run_pb2
from flyteidl2.core import literals_pb2
from flyteidl2.dataproxy import dataproxy_service_pb2
from flyteidl2.task import common_pb2 as task_common_pb2
from flyteidl2.task import run_pb2
from flyteidl2.trigger import trigger_definition_pb2, trigger_service_pb2
from flyteidl2.workflow import run_definition_pb2, run_service_pb2
from mock.mock import AsyncMock, MagicMock

import flyte
from flyte._initialize import _init_for_testing
from flyte._internal.runtime.convert import KICKOFF_TIME_INPUT_ARG_CONTEXT_KEY, Inputs
from flyte.remote._trigger import Trigger as RemoteTrigger
from flyte.remote._trigger import TriggerDetails

TASK_NAME = "env.report"
TASK_VERSION = "v7"


def _mock_client():
    mock_client = MagicMock()
    mock_run_service = AsyncMock()
    mock_run_service.create_run.return_value = run_service_pb2.CreateRunResponse(
        run=run_definition_pb2.Run(
            action=run_definition_pb2.Action(
                id=identifier_pb2.ActionIdentifier(name="a0", run=identifier_pb2.RunIdentifier(name="r-new"))
            )
        )
    )
    mock_client.run_service = mock_run_service
    mock_dataproxy = AsyncMock()
    mock_dataproxy.upload_inputs.return_value = dataproxy_service_pb2.UploadInputsResponse(
        offloaded_input_data=common_run_pb2.OffloadedInputData(uri="s3://b/uploaded/inputs.pb", inputs_hash="h"),
    )
    mock_client.dataproxy_service = mock_dataproxy
    return mock_client, mock_run_service, mock_dataproxy


def _task_details():
    """A fetched TaskDetails stand-in: string input `region`, int input `days`, version v7."""
    from flyteidl2.core import identifier_pb2 as core_identifier_pb2
    from flyteidl2.core import interface_pb2, tasks_pb2, types_pb2
    from flyteidl2.task import task_definition_pb2

    from flyte.remote._task import TaskDetails

    iface = interface_pb2.TypedInterface(
        inputs=interface_pb2.VariableMap(
            variables=[
                interface_pb2.VariableEntry(
                    key="region",
                    value=interface_pb2.Variable(type=types_pb2.LiteralType(simple=types_pb2.SimpleType.STRING)),
                ),
                interface_pb2.VariableEntry(
                    key="days",
                    value=interface_pb2.Variable(type=types_pb2.LiteralType(simple=types_pb2.SimpleType.INTEGER)),
                ),
            ]
        )
    )
    tmpl = tasks_pb2.TaskTemplate(
        id=core_identifier_pb2.Identifier(name=TASK_NAME, version=TASK_VERSION), interface=iface
    )
    pb2 = task_definition_pb2.TaskDetails(
        task_id=task_definition_pb2.TaskIdentifier(
            org="o", project="p", domain="d", name=TASK_NAME, version=TASK_VERSION
        ),
        spec=task_definition_pb2.TaskSpec(task_template=tmpl),
    )
    return TaskDetails(pb2=pb2)


def _str(v: str) -> literals_pb2.Literal:
    return literals_pb2.Literal(scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(string_value=v)))


def _int(v: int) -> literals_pb2.Literal:
    return literals_pb2.Literal(scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(integer=v)))


def _trigger_details(*, inline_inputs=None, offloaded_uri=None, context=(), run_spec=None) -> TriggerDetails:
    spec = trigger_definition_pb2.TriggerSpec(active=True, task_version=TASK_VERSION)
    if inline_inputs is not None:
        spec.inputs.CopyFrom(
            task_common_pb2.Inputs(
                literals=[task_common_pb2.NamedLiteral(name=k, value=v) for k, v in inline_inputs.items()],
                context=[literals_pb2.KeyValuePair(key=k, value=v) for k, v in context],
            )
        )
    if offloaded_uri is not None:
        spec.offloaded_input_data.CopyFrom(common_run_pb2.OffloadedInputData(uri=offloaded_uri, inputs_hash="th"))
    if run_spec is None:
        run_spec = run_pb2.RunSpec(
            envs=run_pb2.Envs(values=[literals_pb2.KeyValuePair(key="REPORT_VERBOSE", value="1")]),
            notification_rules=run_pb2.InlineRuleList(rules=[run_pb2.InlineRule()]),
        )
    spec.run_spec.CopyFrom(run_spec)
    return TriggerDetails(
        pb2=trigger_definition_pb2.TriggerDetails(
            id=identifier_pb2.TriggerIdentifier(
                name=identifier_pb2.TriggerName(
                    org="o", project="p", domain="d", name="full-report", task_name=TASK_NAME
                )
            ),
            spec=spec,
        )
    )


def _literals(inputs: task_common_pb2.Inputs) -> dict:
    return {lit.name: lit.value for lit in inputs.literals}


def _patched_task_get(task_details):
    """Patch remote Task.get to return a lazy entity whose fetch yields `task_details`."""
    lazy = MagicMock()
    lazy.fetch = MagicMock()
    lazy.fetch.aio = AsyncMock(return_value=task_details)
    return mock.patch("flyte.remote._task.Task.get", return_value=lazy)


@pytest.mark.asyncio
async def test_run_trigger_fires_as_trigger_with_registered_inputs_and_runspec():
    """No kwargs + inline inputs: the trigger's literals are uploaded verbatim, the run is created
    with trigger_name (not task_id) and the trigger's run spec incl. notification rules."""
    mock_client, mock_run_service, mock_dataproxy = _mock_client()
    await _init_for_testing(client=mock_client, project="p", domain="d", org="o")
    trigger = _trigger_details(inline_inputs={"region": _str("all"), "days": _int(30)})

    with _patched_task_get(_task_details()) as task_get:
        run = await flyte.run.aio(trigger)

    assert run.name == "r-new"
    task_get.assert_called_once_with(name=TASK_NAME, project="p", domain="d", version=TASK_VERSION)

    upload_req = mock_dataproxy.upload_inputs.call_args[0][0]
    assert upload_req.task_id.name == TASK_NAME and upload_req.task_id.version == TASK_VERSION
    assert _literals(upload_req.inputs) == {"region": _str("all"), "days": _int(30)}

    req: run_service_pb2.CreateRunRequest = mock_run_service.create_run.call_args[0][0]
    assert req.WhichOneof("task") == "trigger_name"
    assert req.trigger_name.name == "full-report" and req.trigger_name.task_name == TASK_NAME
    assert req.trigger_name.project == "p" and req.trigger_name.domain == "d" and req.trigger_name.org == "o"
    assert req.offloaded_input_data.uri == "s3://b/uploaded/inputs.pb"
    # The runner adds its standard env (log levels, sys path); the trigger's env rides along.
    assert {kv.key: kv.value for kv in req.run_spec.envs.values}["REPORT_VERBOSE"] == "1"
    assert len(req.run_spec.notification_rules.rules) == 1


@pytest.mark.asyncio
async def test_run_trigger_offloaded_inputs_without_overrides_reuses_blob():
    """Offloaded inputs + no kwargs: nothing is uploaded; the trigger's blob is referenced as-is."""
    mock_client, mock_run_service, mock_dataproxy = _mock_client()
    await _init_for_testing(client=mock_client, project="p", domain="d", org="o")
    trigger = _trigger_details(offloaded_uri="s3://b/trigger/inputs.pb")

    with _patched_task_get(_task_details()), mock.patch("flyte._internal.runtime.io.load_inputs") as load:
        await flyte.run.aio(trigger)

    load.assert_not_called()
    mock_dataproxy.upload_inputs.assert_not_called()
    req: run_service_pb2.CreateRunRequest = mock_run_service.create_run.call_args[0][0]
    assert req.WhichOneof("task") == "trigger_name"
    assert req.offloaded_input_data.uri == "s3://b/trigger/inputs.pb"
    assert req.offloaded_input_data.inputs_hash == "th"


@pytest.mark.asyncio
async def test_run_trigger_offloaded_inputs_merge_overrides():
    """Offloaded inputs + kwargs: the blob is read back, the override replaces one literal, the
    rest keep the trigger's values, and the merged set is uploaded."""
    mock_client, mock_run_service, mock_dataproxy = _mock_client()
    await _init_for_testing(client=mock_client, project="p", domain="d", org="o")
    trigger = _trigger_details(offloaded_uri="s3://b/trigger/inputs.pb")
    stored = task_common_pb2.Inputs(
        literals=[
            task_common_pb2.NamedLiteral(name="region", value=_str("all")),
            task_common_pb2.NamedLiteral(name="days", value=_int(30)),
        ],
        context=[literals_pb2.KeyValuePair(key="team", value="data")],
    )

    with (
        _patched_task_get(_task_details()),
        mock.patch("flyte._internal.runtime.io.load_inputs", new=AsyncMock(return_value=Inputs(stored))) as load,
    ):
        await flyte.run.aio(trigger, days=7)

    load.assert_awaited_once_with("s3://b/trigger/inputs.pb")
    upload_req = mock_dataproxy.upload_inputs.call_args[0][0]
    assert _literals(upload_req.inputs) == {"region": _str("all"), "days": _int(7)}
    # Trigger-registered context survives the merge.
    assert {kv.key: kv.value for kv in upload_req.inputs.context} == {"team": "data"}
    req: run_service_pb2.CreateRunRequest = mock_run_service.create_run.call_args[0][0]
    assert req.WhichOneof("task") == "trigger_name"


@pytest.mark.asyncio
async def test_run_trigger_override_adds_input_the_trigger_never_bound():
    """An input the trigger left to the task default can still be set by keyword."""
    mock_client, _run_service, mock_dataproxy = _mock_client()
    await _init_for_testing(client=mock_client, project="p", domain="d", org="o")
    trigger = _trigger_details(inline_inputs={"region": _str("all")})

    with _patched_task_get(_task_details()):
        await flyte.run.aio(trigger, days=3)

    upload_req = mock_dataproxy.upload_inputs.call_args[0][0]
    assert _literals(upload_req.inputs) == {"region": _str("all"), "days": _int(3)}


@pytest.mark.asyncio
async def test_run_trigger_artifact_override_binds_stored_literal():
    """A `flyte.remote.Artifact` keyword override binds as the artifact's stored literal, with
    its artifact_id stamp intact, the same way it does for flyte.run(task, ...)."""
    from flyteidl2.artifact import artifact_pb2
    from flyteidl2.core import artifact_id_pb2

    from flyte.remote import Artifact
    from flyte.types import TypeEngine

    mock_client, _run_service, mock_dataproxy = _mock_client()
    await _init_for_testing(client=mock_client, project="p", domain="d", org="o")
    trigger = _trigger_details(inline_inputs={"region": _str("all"), "days": _int(30)})

    version_id = artifact_id_pb2.ArtifactVersionId(
        key=artifact_id_pb2.ArtifactKey(org="o", project="p", domain="d", name="region_pick"), version="3"
    )
    stored = _str("eu")
    stored.artifact_id.CopyFrom(version_id)
    artifact = Artifact(
        pb2=artifact_pb2.Artifact(
            artifact_id=artifact_pb2.ArtifactIdentifier(
                name=artifact_pb2.ArtifactName(org="o", project="p", domain="d", name="region_pick"), version="3"
            ),
            spec=artifact_pb2.ArtifactSpec(value=stored, type=TypeEngine.to_literal_type(str)),
        )
    )

    with _patched_task_get(_task_details()):
        await flyte.run.aio(trigger, region=artifact)

    uploaded = _literals(mock_dataproxy.upload_inputs.call_args[0][0].inputs)
    assert uploaded["days"] == _int(30)
    assert uploaded["region"].scalar.primitive.string_value == "eu"
    assert uploaded["region"].artifact_id == version_id


@pytest.mark.asyncio
async def test_run_trigger_explicit_kickoff_input_drops_marker():
    """Passing the TriggerTime-bound input explicitly removes the kickoff marker so the runtime
    does not overwrite the value with run_start_time; otherwise the marker is kept."""
    mock_client, _run_service, mock_dataproxy = _mock_client()
    await _init_for_testing(client=mock_client, project="p", domain="d", org="o")
    marker = ((KICKOFF_TIME_INPUT_ARG_CONTEXT_KEY, "region"),)

    trigger = _trigger_details(inline_inputs={"days": _int(1)}, context=marker)
    with _patched_task_get(_task_details()):
        await flyte.run.aio(trigger, days=2)
    ctx = {kv.key: kv.value for kv in mock_dataproxy.upload_inputs.call_args[0][0].inputs.context}
    assert ctx == {KICKOFF_TIME_INPUT_ARG_CONTEXT_KEY: "region"}

    with _patched_task_get(_task_details()):
        await flyte.run.aio(trigger, region="explicit")
    ctx = {kv.key: kv.value for kv in mock_dataproxy.upload_inputs.call_args[0][0].inputs.context}
    assert KICKOFF_TIME_INPUT_ARG_CONTEXT_KEY not in ctx


@pytest.mark.asyncio
async def test_run_trigger_runcontext_overrides_layer_on_trigger_runspec():
    """with_runcontext env vars / queue merge on top of the trigger's run spec; the trigger's
    notification rules survive."""
    mock_client, mock_run_service, _dataproxy = _mock_client()
    await _init_for_testing(client=mock_client, project="p", domain="d", org="o")
    trigger = _trigger_details(inline_inputs={"region": _str("all")})

    with _patched_task_get(_task_details()):
        await flyte.with_runcontext(env_vars={"EXTRA": "x"}, queue="gpu", name="named-run").run.aio(trigger)

    req: run_service_pb2.CreateRunRequest = mock_run_service.create_run.call_args[0][0]
    envs = {kv.key: kv.value for kv in req.run_spec.envs.values}
    assert envs["REPORT_VERBOSE"] == "1" and envs["EXTRA"] == "x"
    assert req.run_spec.queue == "gpu"
    assert len(req.run_spec.notification_rules.rules) == 1
    assert req.run_id.name == "named-run"
    assert req.run_id.project == "p" and req.run_id.domain == "d" and req.run_id.org == "o"


@pytest.mark.asyncio
async def test_run_trigger_from_listed_trigger_fetches_details():
    """A `flyte.remote.Trigger` from listall() carries no spec: details are fetched by name."""
    mock_client, mock_run_service, _dataproxy = _mock_client()
    await _init_for_testing(client=mock_client, project="p", domain="d", org="o")
    details = _trigger_details(inline_inputs={"region": _str("all")})
    listed = RemoteTrigger(pb2=details.trigger)
    mock_client.trigger_service = AsyncMock()
    mock_client.trigger_service.get_trigger_details.return_value = trigger_service_pb2.GetTriggerDetailsResponse(
        trigger=details.pb2
    )

    with _patched_task_get(_task_details()):
        await flyte.run.aio(listed)

    get_req = mock_client.trigger_service.get_trigger_details.call_args.kwargs["request"]
    assert get_req.name.name == "full-report" and get_req.name.task_name == TASK_NAME
    req: run_service_pb2.CreateRunRequest = mock_run_service.create_run.call_args[0][0]
    assert req.trigger_name.name == "full-report"


@pytest.mark.asyncio
async def test_run_trigger_rejects_positional_args_and_unknown_inputs():
    mock_client, _run_service, _dataproxy = _mock_client()
    await _init_for_testing(client=mock_client, project="p", domain="d", org="o")
    trigger = _trigger_details(inline_inputs={"region": _str("all")})

    with pytest.raises(ValueError, match="keyword"):
        await flyte.run.aio(trigger, "positional")
    with _patched_task_get(_task_details()), pytest.raises(ValueError, match="Unknown input"):
        await flyte.run.aio(trigger, nope=1)


@pytest.mark.asyncio
async def test_run_trigger_requires_remote_mode():
    mock_client, _run_service, _dataproxy = _mock_client()
    await _init_for_testing(client=mock_client, project="p", domain="d", org="o")
    trigger = _trigger_details(inline_inputs={})
    with pytest.raises(ValueError, match="remote mode"):
        await flyte.with_runcontext(mode="local").run.aio(trigger)
