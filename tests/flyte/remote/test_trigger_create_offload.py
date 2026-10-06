"""Tests for the offloaded-inputs path in flyte.remote.Trigger.create."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.common import identifier_pb2, run_pb2
from flyteidl2.core import interface_pb2, types_pb2
from flyteidl2.core.interface_pb2 import VariableEntry
from flyteidl2.dataproxy import dataproxy_service_pb2
from flyteidl2.task import common_pb2
from flyteidl2.trigger import trigger_definition_pb2, trigger_service_pb2

import flyte
from flyte._internal.runtime.trigger_serde import (
    KICKOFF_TIME_INPUT_ARG_CONTEXT_KEY,
    offload_trigger_inputs,
)
from flyte.remote._trigger import Trigger


def _task_details(version: str = "v1"):
    """A minimal stand-in for remote.TaskDetails with the fields create() reads."""
    task_inputs = interface_pb2.VariableMap(
        variables=[
            VariableEntry(
                key="start_time",
                value=interface_pb2.Variable(type=types_pb2.LiteralType(simple=types_pb2.SimpleType.DATETIME)),
            ),
            VariableEntry(
                key="x",
                value=interface_pb2.Variable(type=types_pb2.LiteralType(simple=types_pb2.SimpleType.INTEGER)),
            ),
        ]
    )
    details = MagicMock()
    details.version = version
    details.pb2.spec.task_template.interface.inputs = task_inputs
    details.pb2.spec.default_inputs = []
    return details


def _client(existing_revision=None, existing_active=None, lookup_error=None, conflict_code=None):
    """
    A client mock; `existing_revision` makes a trigger of the same name already exist at that revision, so a deploy
    with any other revision fails as the backend's optimistic lock does (FAILED_PRECONDITION, or `conflict_code`).
    """
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    client = MagicMock()
    if lookup_error is not None:
        client.trigger_service.get_trigger_details = AsyncMock(side_effect=lookup_error)
    elif existing_revision is None:
        client.trigger_service.get_trigger_details = AsyncMock(side_effect=ConnectError(Code.NOT_FOUND, "no trigger"))
    else:
        client.trigger_service.get_trigger_details = AsyncMock(
            return_value=trigger_service_pb2.GetTriggerDetailsResponse(
                trigger=trigger_definition_pb2.TriggerDetails(
                    id=identifier_pb2.TriggerIdentifier(
                        name=identifier_pb2.TriggerName(name="t", task_name="my_task"), revision=existing_revision
                    ),
                    spec=(
                        trigger_definition_pb2.TriggerSpec(active=existing_active)
                        if existing_active is not None
                        else None
                    ),
                )
            )
        )

    async def _deploy(request):
        if existing_revision is not None and request.revision != existing_revision:
            raise ConnectError(conflict_code or Code.FAILED_PRECONDITION, "optimistic lock failure")
        return trigger_service_pb2.DeployTriggerResponse(trigger=trigger_definition_pb2.TriggerDetails())

    client.trigger_service.deploy_trigger = AsyncMock(side_effect=_deploy)
    return client


@pytest.mark.asyncio
async def test_create_offloads_inputs_and_stores_uri():
    cfg = MagicMock(org="o", project="p", domain="d")

    offloaded = run_pb2.OffloadedInputData(uri="s3://bucket/offloaded-inputs/abc/inputs.pb", inputs_hash="abc")
    client = _client()
    client.dataproxy_service.upload_trigger = AsyncMock(
        return_value=dataproxy_service_pb2.UploadInputsResponse(offloaded_input_data=offloaded)
    )
    client.trigger_service.deploy_trigger = AsyncMock(
        return_value=trigger_service_pb2.DeployTriggerResponse(
            trigger=trigger_definition_pb2.TriggerDetails(
                id=identifier_pb2.TriggerIdentifier(
                    name=identifier_pb2.TriggerName(name="t", task_name="my_task", org="o", project="p", domain="d")
                )
            )
        )
    )

    lazy = MagicMock()
    lazy.fetch.aio = AsyncMock(return_value=_task_details(version="v1"))

    trigger = flyte.Trigger(
        name="t",
        automation=flyte.Cron("0 0 * * *"),
        inputs={"start_time": flyte.TriggerTime, "x": 7},
    )

    with (
        patch("flyte.remote._trigger.ensure_client"),
        patch("flyte.remote._trigger.get_init_config", return_value=cfg),
        patch("flyte.remote._trigger.get_client", return_value=client),
        # offload_trigger_inputs (in trigger_serde) resolves the client via flyte._initialize.
        patch("flyte._initialize.get_client", return_value=client),
        patch("flyte.remote._trigger.Task.get", return_value=lazy),
    ):
        await Trigger.create.aio(trigger, task_name="my_task")

    # 1) upload_trigger was called before deploy, targeting the task (not the not-yet-existent trigger).
    client.dataproxy_service.upload_trigger.assert_awaited_once()
    upload_req = client.dataproxy_service.upload_trigger.await_args[0][0]
    assert upload_req.WhichOneof("task") == "task_id"
    assert upload_req.task_id.name == "my_task"
    assert upload_req.task_id.version == "v1"
    assert upload_req.project_id.name == "p"
    # The kickoff arg name rides along in the offloaded inputs context.
    ctx = {kv.key: kv.value for kv in upload_req.inputs.context}
    assert ctx[KICKOFF_TIME_INPUT_ARG_CONTEXT_KEY] == "start_time"
    # The non-TriggerTime default input is offloaded as a literal.
    lit_names = {lit.name for lit in upload_req.inputs.literals}
    assert "x" in lit_names
    assert "start_time" not in lit_names  # kickoff arg is not offloaded as a literal

    # 2) deploy_trigger stored the offloaded URI and did NOT set inline inputs.
    deploy_req = client.trigger_service.deploy_trigger.await_args.kwargs["request"]
    spec = deploy_req.spec
    assert spec.WhichOneof("input_wrapper") == "offloaded_input_data"
    assert spec.offloaded_input_data.uri == offloaded.uri
    assert spec.offloaded_input_data.inputs_hash == "abc"
    assert spec.task_version == "v1"
    # The schedule carries the kickoff arg; the name is also conveyed via the offloaded inputs context.
    assert deploy_req.automation_spec.schedule.kickoff_time_input_arg == "start_time"


@pytest.mark.asyncio
async def test_offload_trigger_inputs_uses_task_spec_for_deploy_path():
    """Deploy path references the not-yet-registered task by task_spec (no server lookup)."""
    from flyteidl2.task import task_definition_pb2

    offloaded = run_pb2.OffloadedInputData(uri="s3://bucket/x/inputs.pb", inputs_hash="h")
    client = _client()
    client.dataproxy_service.upload_trigger = AsyncMock(
        return_value=dataproxy_service_pb2.UploadInputsResponse(offloaded_input_data=offloaded)
    )

    spec = task_definition_pb2.TaskSpec()
    inputs = common_pb2.Inputs()

    with patch("flyte._initialize.get_client", return_value=client):
        out = await offload_trigger_inputs(inputs, org="o", project="p", domain="d", task_version="v1", task_spec=spec)

    assert out == offloaded
    req = client.dataproxy_service.upload_trigger.await_args[0][0]
    assert req.WhichOneof("task") == "task_spec"
    assert req.WhichOneof("id") == "project_id"
    assert req.project_id.name == "p"


@pytest.mark.asyncio
async def test_offload_trigger_inputs_uses_task_id_when_named():
    offloaded = run_pb2.OffloadedInputData(uri="s3://bucket/x/inputs.pb", inputs_hash="h")
    client = _client()
    client.dataproxy_service.upload_trigger = AsyncMock(
        return_value=dataproxy_service_pb2.UploadInputsResponse(offloaded_input_data=offloaded)
    )

    with patch("flyte._initialize.get_client", return_value=client):
        await offload_trigger_inputs(
            common_pb2.Inputs(), org="o", project="p", domain="d", task_version="v1", task_name="my_task"
        )

    req = client.dataproxy_service.upload_trigger.await_args[0][0]
    assert req.WhichOneof("task") == "task_id"
    assert req.task_id.name == "my_task"
    assert req.task_id.version == "v1"


@pytest.mark.asyncio
async def test_offload_trigger_inputs_requires_task_reference():
    with pytest.raises(ValueError, match="task_spec or task_name"):
        await offload_trigger_inputs(common_pb2.Inputs(), org="o", project="p", domain="d", task_version="v1")


@pytest.mark.asyncio
async def test_offload_trigger_inputs_returns_none_on_unimplemented():
    """Zero trust off: SelectCluster returns UNIMPLEMENTED for OPERATION_UPLOAD_TRIGGER -> None."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    client = _client()
    client.dataproxy_service.upload_trigger = AsyncMock(side_effect=ConnectError(Code.UNIMPLEMENTED, "no zero trust"))

    with patch("flyte._initialize.get_client", return_value=client):
        out = await offload_trigger_inputs(
            common_pb2.Inputs(), org="o", project="p", domain="d", task_version="v1", task_name="my_task"
        )

    assert out is None


@pytest.mark.asyncio
async def test_offload_trigger_inputs_reraises_other_connect_errors():
    """Non-UNIMPLEMENTED errors are not swallowed."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    client = _client()
    client.dataproxy_service.upload_trigger = AsyncMock(side_effect=ConnectError(Code.INTERNAL, "boom"))

    with patch("flyte._initialize.get_client", return_value=client):
        with pytest.raises(ConnectError):
            await offload_trigger_inputs(
                common_pb2.Inputs(), org="o", project="p", domain="d", task_version="v1", task_name="my_task"
            )


@pytest.mark.asyncio
async def test_create_falls_back_to_inline_inputs_when_unimplemented():
    """When offload is unavailable, Trigger.create registers inline inputs instead of an offloaded URI."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    cfg = MagicMock(org="o", project="p", domain="d")
    client = _client()
    client.dataproxy_service.upload_trigger = AsyncMock(side_effect=ConnectError(Code.UNIMPLEMENTED, "no zero trust"))
    client.trigger_service.deploy_trigger = AsyncMock(
        return_value=trigger_service_pb2.DeployTriggerResponse(
            trigger=trigger_definition_pb2.TriggerDetails(
                id=identifier_pb2.TriggerIdentifier(
                    name=identifier_pb2.TriggerName(name="t", task_name="my_task", org="o", project="p", domain="d")
                )
            )
        )
    )

    lazy = MagicMock()
    lazy.fetch.aio = AsyncMock(return_value=_task_details(version="v1"))

    trigger = flyte.Trigger(
        name="t",
        automation=flyte.Cron("0 0 * * *"),
        inputs={"start_time": flyte.TriggerTime, "x": 7},
    )

    with (
        patch("flyte.remote._trigger.ensure_client"),
        patch("flyte.remote._trigger.get_init_config", return_value=cfg),
        patch("flyte.remote._trigger.get_client", return_value=client),
        patch("flyte._initialize.get_client", return_value=client),
        patch("flyte.remote._trigger.Task.get", return_value=lazy),
    ):
        await Trigger.create.aio(trigger, task_name="my_task")

    deploy_req = client.trigger_service.deploy_trigger.await_args.kwargs["request"]
    spec = deploy_req.spec
    # Inline inputs, not an offloaded URI.
    assert spec.WhichOneof("input_wrapper") == "inputs"
    lit_names = {lit.name for lit in spec.inputs.literals}
    assert "x" in lit_names


async def _deploy_request(client, auto_activate=True, **kwargs):
    cfg = MagicMock(org="o", project="p", domain="d")
    client.dataproxy_service.upload_trigger = AsyncMock(
        return_value=dataproxy_service_pb2.UploadInputsResponse(
            offloaded_input_data=run_pb2.OffloadedInputData(uri="s3://b/i.pb", inputs_hash="h")
        )
    )
    lazy = MagicMock()
    lazy.fetch.aio = AsyncMock(return_value=_task_details())
    trigger = flyte.Trigger(
        name="t",
        automation=flyte.Cron("0 0 * * *"),
        inputs={"start_time": flyte.TriggerTime},
        auto_activate=auto_activate,
    )
    with (
        patch("flyte.remote._trigger.ensure_client"),
        patch("flyte.remote._trigger.get_init_config", return_value=cfg),
        patch("flyte.remote._trigger.get_client", return_value=client),
        patch("flyte._initialize.get_client", return_value=client),
        patch("flyte.remote._trigger.Task.get", return_value=lazy) as task_get,
    ):
        await Trigger.create.aio(trigger, task_name="my_task", **kwargs)
    client.task_get = task_get
    return client.trigger_service.deploy_trigger.await_args.kwargs["request"]


@pytest.mark.asyncio
async def test_create_new_trigger_is_one_rpc_without_a_revision():
    """A plain create makes exactly one trigger RPC (no prior lookup, no revision), as before revisions."""
    client = _client()
    req = await _deploy_request(client)
    client.trigger_service.get_trigger_details.assert_not_awaited()
    client.trigger_service.deploy_trigger.assert_awaited_once()
    assert req.revision == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["FAILED_PRECONDITION", "ALREADY_EXISTS", "ABORTED"])
async def test_create_replaces_an_existing_trigger_at_its_latest_revision(code):
    """A revision conflict: look up the trigger's latest revision and retry once with it (optimistic locking)."""
    from connectrpc.code import Code

    client = _client(existing_revision=4, conflict_code=getattr(Code, code))
    req = await _deploy_request(client)
    assert client.trigger_service.deploy_trigger.await_count == 2
    first = client.trigger_service.deploy_trigger.await_args_list[0].kwargs["request"]
    assert first.revision == 0
    client.trigger_service.get_trigger_details.assert_awaited_once()
    assert req.revision == 4
    assert req.name.name == "t" and req.name.task_name == "my_task"


@pytest.mark.asyncio
async def test_create_other_deploy_errors_are_not_retried():
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    client = _client()
    client.trigger_service.deploy_trigger = AsyncMock(side_effect=ConnectError(Code.INVALID_ARGUMENT, "bad cron"))
    with pytest.raises(ConnectError) as exc:
        await _deploy_request(client)
    assert exc.value.code == Code.INVALID_ARGUMENT
    client.trigger_service.deploy_trigger.assert_awaited_once()
    client.trigger_service.get_trigger_details.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        pytest.param("connect", id="connect-unavailable"),
        pytest.param(RuntimeError("boom"), id="other-exception"),
    ],
)
async def test_create_lookup_failure_after_conflict_raises_the_conflict(error):
    """The revision lookup after a conflict fails: the original conflict surfaces, with no further deploy."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    if error == "connect":
        error = ConnectError(Code.UNAVAILABLE, "down")
    client = _client(lookup_error=error)
    client.trigger_service.deploy_trigger = AsyncMock(side_effect=ConnectError(Code.FAILED_PRECONDITION, "stale"))
    with pytest.raises(ConnectError) as exc:
        await _deploy_request(client)
    assert exc.value.code == Code.FAILED_PRECONDITION
    client.trigger_service.deploy_trigger.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_retry_conflict_is_not_retried_again():
    """Exactly one retry: a second conflict (a concurrent deploy won the race) is raised."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    client = _client(existing_revision=4)
    client.trigger_service.deploy_trigger = AsyncMock(side_effect=ConnectError(Code.FAILED_PRECONDITION, "stale"))
    with pytest.raises(ConnectError):
        await _deploy_request(client)
    assert client.trigger_service.deploy_trigger.await_count == 2


@pytest.mark.asyncio
async def test_create_new_trigger_uses_auto_activate():
    req = await _deploy_request(_client(), auto_activate=False)
    assert req.revision == 0
    assert req.spec.active is False


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_active", [True, False])
@pytest.mark.parametrize("auto_activate", [True, False])
async def test_create_replacing_applies_the_declared_state(existing_active, auto_activate):
    """Declarative: the deployed trigger's state is its declaration's (auto_activate), on replace too."""
    req = await _deploy_request(
        _client(existing_revision=3, existing_active=existing_active), auto_activate=auto_activate
    )
    assert req.revision == 3
    assert req.spec.active is auto_activate


@pytest.mark.asyncio
@pytest.mark.parametrize("active", [True, False])
async def test_create_explicit_active_overrides_existing_state(active):
    req = await _deploy_request(_client(existing_revision=3, existing_active=not active), active=active)
    assert req.spec.active is active


@pytest.mark.asyncio
async def test_create_uses_explicit_project_and_domain():
    client = _client(existing_revision=2)
    req = await _deploy_request(client, project="other", domain="prod")
    assert client.task_get.call_args.kwargs["project"] == "other"
    assert client.task_get.call_args.kwargs["domain"] == "prod"
    lookup = client.trigger_service.get_trigger_details.await_args.kwargs["request"]
    assert (lookup.name.project, lookup.name.domain) == ("other", "prod")
    assert (req.name.project, req.name.domain) == ("other", "prod")
