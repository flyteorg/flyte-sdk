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


@pytest.mark.asyncio
async def test_create_offloads_inputs_and_stores_uri():
    cfg = MagicMock(org="o", project="p", domain="d")

    offloaded = run_pb2.OffloadedInputData(uri="s3://bucket/offloaded-inputs/abc/inputs.pb", inputs_hash="abc")
    client = MagicMock()
    client.dataproxy_service.upload_inputs = AsyncMock(
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

    # 1) upload_inputs was called before deploy, targeting the task (not the not-yet-existent trigger).
    client.dataproxy_service.upload_inputs.assert_awaited_once()
    upload_req = client.dataproxy_service.upload_inputs.await_args[0][0]
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
    client = MagicMock()
    client.dataproxy_service.upload_inputs = AsyncMock(
        return_value=dataproxy_service_pb2.UploadInputsResponse(offloaded_input_data=offloaded)
    )

    spec = task_definition_pb2.TaskSpec()
    inputs = common_pb2.Inputs()

    with patch("flyte._initialize.get_client", return_value=client):
        out = await offload_trigger_inputs(inputs, org="o", project="p", domain="d", task_version="v1", task_spec=spec)

    assert out == offloaded
    req = client.dataproxy_service.upload_inputs.await_args[0][0]
    assert req.WhichOneof("task") == "task_spec"
    assert req.WhichOneof("id") == "project_id"
    assert req.project_id.name == "p"


@pytest.mark.asyncio
async def test_offload_trigger_inputs_uses_task_id_when_named():
    offloaded = run_pb2.OffloadedInputData(uri="s3://bucket/x/inputs.pb", inputs_hash="h")
    client = MagicMock()
    client.dataproxy_service.upload_inputs = AsyncMock(
        return_value=dataproxy_service_pb2.UploadInputsResponse(offloaded_input_data=offloaded)
    )

    with patch("flyte._initialize.get_client", return_value=client):
        await offload_trigger_inputs(
            common_pb2.Inputs(), org="o", project="p", domain="d", task_version="v1", task_name="my_task"
        )

    req = client.dataproxy_service.upload_inputs.await_args[0][0]
    assert req.WhichOneof("task") == "task_id"
    assert req.task_id.name == "my_task"
    assert req.task_id.version == "v1"


@pytest.mark.asyncio
async def test_offload_trigger_inputs_requires_task_reference():
    with pytest.raises(ValueError, match="task_spec or task_name"):
        await offload_trigger_inputs(common_pb2.Inputs(), org="o", project="p", domain="d", task_version="v1")
