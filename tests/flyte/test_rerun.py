"""Unit tests for flyte.rerun (folded into _Runner): re-run a prior run by fetching its
RunSpec + task spec + inputs and resubmitting via the shared _submit_remote path."""

from types import SimpleNamespace

import mock
import pytest
from flyteidl2.common import run_pb2 as common_run_pb2
from flyteidl2.core import literals_pb2
from flyteidl2.dataproxy import dataproxy_service_pb2
from flyteidl2.task import common_pb2 as task_common_pb2
from flyteidl2.task import run_pb2
from flyteidl2.workflow import run_definition_pb2, run_service_pb2
from mock.mock import AsyncMock, MagicMock

import flyte
from flyte._initialize import _init_for_testing

# RunSpec.relation (provenance: related_to + relation_type) ships in a flyteidl2 release newer
# than the current pin; relation-bearing assertions are gated on the installed build.
_RELATION_SUPPORTED = "relation" in run_pb2.RunSpec.DESCRIPTOR.fields_by_name

needs_relation = pytest.mark.skipif(not _RELATION_SUPPORTED, reason="flyteidl2 build lacks RunSpec.relation")


def _mock_client_with_run():
    """Mock client whose create_run captures the request and get_action_data returns prior inputs."""
    mock_client = MagicMock()
    mock_run_service = AsyncMock()
    mock_client.run_service = mock_run_service

    mock_dataproxy = AsyncMock()
    mock_dataproxy.upload_inputs.return_value = dataproxy_service_pb2.UploadInputsResponse(
        offloaded_input_data=common_run_pb2.OffloadedInputData(uri="s3://b/inputs", inputs_hash="h"),
    )
    # Prior run's raw proto inputs (what get_action_data returns).
    prior_inputs = task_common_pb2.Inputs(
        literals=[
            task_common_pb2.NamedLiteral(
                name="v",
                value=literals_pb2.Literal(
                    scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(string_value="prior"))
                ),
            )
        ]
    )
    mock_dataproxy.get_action_data.return_value = dataproxy_service_pb2.GetActionDataResponse(inputs=prior_inputs)
    mock_client.dataproxy_service = mock_dataproxy
    return mock_client, mock_run_service, mock_dataproxy, prior_inputs


def _fake_prior_run(base_envs=None, action_id=None, base_run_spec=None):
    """A stand-in RunDetails: prior RunSpec + a root action carrying a task spec.

    ``action_id`` optionally carries the prior run's full ActionIdentifier (as the real
    fetch would); ``base_run_spec`` optionally substitutes the prior RunSpec wholesale.
    """
    if base_run_spec is None:
        base_run_spec = run_pb2.RunSpec(
            envs=run_pb2.Envs(values=[literals_pb2.KeyValuePair(key="KEEP", value="1")] + (base_envs or [])),
            queue="orig",
        )
    task_spec = run_definition_pb2.ActionDetails(
        id=action_id,
        task=_task_spec_with_string_input(),
    )
    action_details = SimpleNamespace(pb2=task_spec)
    run_details = SimpleNamespace(
        pb2=SimpleNamespace(run_spec=base_run_spec),
        action_details=action_details,
    )
    return run_details


def _task_spec_with_string_input():
    """A minimal TaskSpec with one string input `v` and a version, for fetch + guess_interface."""
    from flyteidl2.core import identifier_pb2, interface_pb2, tasks_pb2, types_pb2
    from flyteidl2.task import task_definition_pb2

    iface = interface_pb2.TypedInterface(
        inputs=interface_pb2.VariableMap(
            variables=[
                interface_pb2.VariableEntry(
                    key="v",
                    value=interface_pb2.Variable(type=types_pb2.LiteralType(simple=types_pb2.SimpleType.STRING)),
                )
            ]
        )
    )
    tmpl = tasks_pb2.TaskTemplate(
        id=identifier_pb2.Identifier(name="test.task1", version="v1"),
        interface=iface,
    )
    return task_definition_pb2.TaskSpec(task_template=tmpl)


def _string_inputs(**values) -> task_common_pb2.Inputs:
    """A prior-run Inputs proto with one string literal per keyword."""
    return task_common_pb2.Inputs(
        literals=[
            task_common_pb2.NamedLiteral(
                name=name,
                value=literals_pb2.Literal(
                    scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(string_value=value))
                ),
            )
            for name, value in values.items()
        ]
    )


def _add_string_input(iface, name: str):
    """Append another string input to a TypedInterface built by _task_spec_with_string_input."""
    from flyteidl2.core import interface_pb2, types_pb2

    iface.inputs.variables.append(
        interface_pb2.VariableEntry(
            key=name,
            value=interface_pb2.Variable(type=types_pb2.LiteralType(simple=types_pb2.SimpleType.STRING)),
        )
    )


@pytest.mark.asyncio
async def test_rerun_same_inputs_inherits_runspec_and_reuses_prior_inputs():
    mock_client, mock_run_service, mock_dataproxy, prior_inputs = _mock_client_with_run()
    await _init_for_testing(client=mock_client, project="test", domain="test")

    with mock.patch("flyte.remote._run.RunDetails") as RD:
        RD.get.aio = AsyncMock(return_value=_fake_prior_run())
        run = await flyte.with_runcontext(mode="remote", env_vars={"X": "1"}).rerun.aio("r1")

    assert run
    # Prior inputs reused verbatim (no conversion).
    mock_dataproxy.get_action_data.assert_called_once()
    upload_req = mock_dataproxy.upload_inputs.call_args[0][0]
    assert upload_req.inputs == prior_inputs

    req: run_service_pb2.CreateRunRequest = mock_run_service.create_run.call_args[0][0]
    envs = {kv.key: kv.value for kv in req.run_spec.envs.values}
    assert envs["KEEP"] == "1"  # inherited from prior run
    assert envs["X"] == "1"  # runner override merged in
    assert req.run_spec.queue == "orig"  # inherited (queue not overridden)
    assert req.WhichOneof("task") == "task_spec"
    assert req.task_spec.task_template.id.name == "test.task1"


@pytest.mark.asyncio
async def test_rerun_changed_inputs_converts_against_fetched_interface():
    mock_client, _mock_run_service, mock_dataproxy, _ = _mock_client_with_run()
    await _init_for_testing(client=mock_client, project="test", domain="test")

    with mock.patch("flyte.remote._run.RunDetails") as RD:
        RD.get.aio = AsyncMock(return_value=_fake_prior_run())
        run = await flyte.with_runcontext(mode="remote").rerun.aio("r1", v="changed")

    assert run
    # Changed inputs => no prior-input fetch; converted against the fetched interface.
    mock_dataproxy.get_action_data.assert_not_called()
    upload_req = mock_dataproxy.upload_inputs.call_args[0][0]
    assert upload_req.inputs.literals[0].name == "v"
    assert upload_req.inputs.literals[0].value.scalar.primitive.string_value == "changed"


@pytest.mark.asyncio
async def test_rerun_partial_inputs_merge_with_the_source_run_inputs():
    """Changing one input keeps the source run's value for every input left out."""
    mock_client, _mock_run_service, mock_dataproxy, _ = _mock_client_with_run()
    await _init_for_testing(client=mock_client, project="test", domain="test")

    prior = _fake_prior_run()
    _add_string_input(prior.action_details.pb2.task.task_template.interface, "w")
    mock_dataproxy.get_action_data.return_value = dataproxy_service_pb2.GetActionDataResponse(
        inputs=_string_inputs(v="prior-v", w="prior-w")
    )

    with mock.patch("flyte.remote._run.RunDetails") as RD:
        RD.get.aio = AsyncMock(return_value=prior)
        await flyte.with_runcontext(mode="remote").rerun.aio("r1", w="changed")

    # The unchanged input can only come from the source run, so it is fetched.
    mock_dataproxy.get_action_data.assert_called_once()
    upload_req = mock_dataproxy.upload_inputs.call_args[0][0]
    literals = {lit.name: lit.value.scalar.primitive.string_value for lit in upload_req.inputs.literals}
    assert literals == {"v": "prior-v", "w": "changed"}


@pytest.mark.asyncio
async def test_rerun_full_input_set_skips_the_source_input_fetch():
    """Covering every input is the escape hatch when the source inputs are gone."""
    mock_client, _mock_run_service, mock_dataproxy, _ = _mock_client_with_run()
    await _init_for_testing(client=mock_client, project="test", domain="test")

    prior = _fake_prior_run()
    _add_string_input(prior.action_details.pb2.task.task_template.interface, "w")

    with mock.patch("flyte.remote._run.RunDetails") as RD:
        RD.get.aio = AsyncMock(return_value=prior)
        await flyte.with_runcontext(mode="remote").rerun.aio("r1", v="a", w="b")

    mock_dataproxy.get_action_data.assert_not_called()
    upload_req = mock_dataproxy.upload_inputs.call_args[0][0]
    literals = {lit.name: lit.value.scalar.primitive.string_value for lit in upload_req.inputs.literals}
    assert literals == {"v": "a", "w": "b"}


@pytest.mark.asyncio
async def test_rerun_rejects_unknown_input_names():
    """A typo would otherwise be silently dropped from the merged inputs."""
    mock_client, _mock_run_service, _mock_dataproxy, _ = _mock_client_with_run()
    await _init_for_testing(client=mock_client, project="test", domain="test")

    with mock.patch("flyte.remote._run.RunDetails") as RD:
        RD.get.aio = AsyncMock(return_value=_fake_prior_run())
        with pytest.raises(ValueError, match=r"Unknown input\(s\) \['nope'\].*Known inputs: v"):
            await flyte.with_runcontext(mode="remote").rerun.aio("r1", nope="x")


@pytest.mark.asyncio
async def test_partial_inputs_need_readable_source_inputs():
    """With only the source inputs' URI in hand there is nothing to merge into, so the partial
    change fails loudly instead of dropping the inputs that were meant to be kept."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    import flyte.errors

    mock_client, mock_run_service, mock_dataproxy, _ = _mock_client_with_run()
    mock_dataproxy.get_action_data.side_effect = ConnectError(
        Code.NOT_FOUND, "object 's3://b/metadata/v2/p/d/r1/a0/1/outputs.pb' not found"
    )
    mock_run_service.get_action_data_u_r_is.return_value = run_service_pb2.GetActionDataURIsResponse(
        inputs_uri="s3://b/metadata/v2/p/d/r1/a0/inputs.pb", outputs_uri=""
    )
    await _init_for_testing(client=mock_client, project="test", domain="test")

    prior = _fake_prior_run()
    _add_string_input(prior.action_details.pb2.task.task_template.interface, "w")

    with mock.patch("flyte.remote._run.RunDetails") as RD:
        RD.get.aio = AsyncMock(return_value=prior)
        with pytest.raises(flyte.errors.RuntimeUserError, match="cannot be merged"):
            await flyte.with_runcontext(mode="remote").rerun.aio("r1", allow_missing_source_outputs=True, w="changed")


@needs_relation
@pytest.mark.asyncio
async def test_recover_with_partial_inputs_merges_and_still_recovers():
    mock_client, mock_run_service, mock_dataproxy, _ = _mock_client_with_run()
    await _init_for_testing(client=mock_client, project="test", domain="test")

    prior = _fake_prior_run()
    _add_string_input(prior.action_details.pb2.task.task_template.interface, "w")
    mock_dataproxy.get_action_data.return_value = dataproxy_service_pb2.GetActionDataResponse(
        inputs=_string_inputs(v="prior-v", w="prior-w")
    )

    with mock.patch("flyte.remote._run.RunDetails") as RD:
        RD.get.aio = AsyncMock(return_value=prior)
        await flyte.with_runcontext(mode="remote").rerun.aio("r1", recover=True, w="changed")

    upload_req = mock_dataproxy.upload_inputs.call_args[0][0]
    literals = {lit.name: lit.value.scalar.primitive.string_value for lit in upload_req.inputs.literals}
    assert literals == {"v": "prior-v", "w": "changed"}
    req: run_service_pb2.CreateRunRequest = mock_run_service.create_run.call_args[0][0]
    assert req.run_spec.relation.relation_type == common_run_pb2.RELATION_TYPE_RECOVER


@needs_relation
@pytest.mark.asyncio
async def test_recover_with_changed_inputs_uses_new_inputs_and_recover_relation():
    """`--recover` + new inputs: the new run starts from the changed inputs (no prior-input
    fetch) and still carries RECOVER provenance so succeeded actions are reused."""
    mock_client, mock_run_service, mock_dataproxy, _ = _mock_client_with_run()
    await _init_for_testing(client=mock_client, project="test", domain="test")

    with mock.patch("flyte.remote._run.RunDetails") as RD:
        RD.get.aio = AsyncMock(return_value=_fake_prior_run())
        run = await flyte.with_runcontext(mode="remote").rerun.aio("r1", recover=True, v="changed")

    assert run
    mock_dataproxy.get_action_data.assert_not_called()
    upload_req = mock_dataproxy.upload_inputs.call_args[0][0]
    assert upload_req.inputs.literals[0].value.scalar.primitive.string_value == "changed"

    req: run_service_pb2.CreateRunRequest = mock_run_service.create_run.call_args[0][0]
    assert req.run_spec.relation.relation_type == common_run_pb2.RELATION_TYPE_RECOVER
    assert req.run_spec.relation.related_to.name == "r1"


@needs_relation
@pytest.mark.asyncio
async def test_recover_with_changed_inputs_keeps_force_rerun_actions():
    """Forcing actions is how a recovery is made to re-execute against the changed inputs."""
    mock_client, mock_run_service, _mock_dataproxy, _ = _mock_client_with_run()
    await _init_for_testing(client=mock_client, project="test", domain="test")

    with mock.patch("flyte.remote._run.RunDetails") as RD:
        RD.get.aio = AsyncMock(return_value=_fake_prior_run())
        await flyte.with_runcontext(mode="remote").rerun.aio(
            "r1", recover=True, force_rerun_actions=["a3"], v="changed"
        )

    req: run_service_pb2.CreateRunRequest = mock_run_service.create_run.call_args[0][0]
    assert list(req.run_spec.recover.force_rerun_actions) == ["a3"]


@needs_relation
@pytest.mark.asyncio
async def test_recover_with_changed_inputs_warns_about_reused_outputs():
    """Recovered actions keep the outputs they produced under the *original* inputs, so the
    combination is warned about rather than silently accepted."""
    mock_client, _mock_run_service, _mock_dataproxy, _ = _mock_client_with_run()
    await _init_for_testing(client=mock_client, project="test", domain="test")

    with mock.patch("flyte.remote._run.RunDetails") as RD, mock.patch("flyte._run.logger") as log:
        RD.get.aio = AsyncMock(return_value=_fake_prior_run())
        await flyte.with_runcontext(mode="remote").rerun.aio("r1", recover=True, v="changed")

    warned = " ".join(str(c.args[0]) for c in log.warning.call_args_list)
    assert "keeps the output it produced under the original inputs" in warned
    assert "force_rerun_actions" in warned


@needs_relation
@pytest.mark.asyncio
async def test_rerun_records_rerun_relation_and_clears_inherited_provenance():
    from flyteidl2.common import identifier_pb2

    mock_client, mock_run_service, _mock_dataproxy, _ = _mock_client_with_run()
    await _init_for_testing(client=mock_client, project="test", domain="test")

    # The prior run was itself derived from a grandparent; that link must not be inherited.
    prior = _fake_prior_run()
    prior.pb2.run_spec.relation.CopyFrom(
        common_run_pb2.Relation(
            related_to=identifier_pb2.RunIdentifier(name="grandparent"),
            relation_type=common_run_pb2.RELATION_TYPE_RERUN,
        )
    )

    with mock.patch("flyte.remote._run.RunDetails") as RD:
        RD.get.aio = AsyncMock(return_value=prior)
        await flyte.with_runcontext(mode="remote").rerun.aio("r1")

    req: run_service_pb2.CreateRunRequest = mock_run_service.create_run.call_args[0][0]
    assert req.run_spec.HasField("relation")
    assert req.run_spec.relation.related_to.name == "r1"
    # The identifier must be fully qualified (server validates org/project/domain min_len=1).
    assert req.run_spec.relation.related_to.project == "test"
    assert req.run_spec.relation.related_to.domain == "test"
    assert req.run_spec.relation.relation_type == common_run_pb2.RELATION_TYPE_RERUN


@needs_relation
@pytest.mark.asyncio
async def test_rerun_with_recover_records_recover_relation():
    mock_client, mock_run_service, _mock_dataproxy, _ = _mock_client_with_run()
    await _init_for_testing(client=mock_client, project="test", domain="test")

    with mock.patch("flyte.remote._run.RunDetails") as RD:
        RD.get.aio = AsyncMock(return_value=_fake_prior_run())
        await flyte.with_runcontext(mode="remote").rerun.aio("r1", recover=True)

    req: run_service_pb2.CreateRunRequest = mock_run_service.create_run.call_args[0][0]
    assert req.run_spec.relation.related_to.name == "r1"
    assert req.run_spec.relation.related_to.project == "test"
    assert req.run_spec.relation.related_to.domain == "test"
    assert req.run_spec.relation.relation_type == common_run_pb2.RELATION_TYPE_RECOVER


@pytest.mark.skipif(_RELATION_SUPPORTED, reason="only applies to flyteidl2 builds without RunSpec.relation")
@pytest.mark.asyncio
async def test_recover_raises_without_relation_field():
    """On old flyteidl2, plain rerun still works (provenance skipped) but recover fails loudly."""
    mock_client, _mock_run_service, _mock_dataproxy, _ = _mock_client_with_run()
    await _init_for_testing(client=mock_client, project="test", domain="test")

    with mock.patch("flyte.remote._run.RunDetails") as RD:
        RD.get.aio = AsyncMock(return_value=_fake_prior_run())
        with pytest.raises(NotImplementedError, match="recover is not yet supported"):
            await flyte.with_runcontext(mode="remote").rerun.aio("r1", recover=True)


@pytest.mark.asyncio
async def test_rerun_rejects_non_remote_mode():
    await flyte.init.aio()
    with pytest.raises(NotImplementedError, match="remote mode"):
        await flyte.with_runcontext(mode="local").rerun.aio("r1")


def test_replay_is_removed():
    """flyte.replay was deleted in favor of flyte.rerun."""
    assert not hasattr(flyte, "replay")


# Provenance for rerun/recover is asserted on RunSpec.relation by
# test_rerun_records_rerun_relation_and_clears_inherited_provenance and
# test_rerun_with_recover_records_recover_relation (both gated on flyteidl2 shipping the field).
# The former related_to-based rerun tests were removed with that migration.


@pytest.mark.asyncio
async def test_rerun_missing_source_outputs_opt_in_falls_back_to_inputs_uri():
    """With allow_missing_source_outputs, deleted source outputs fall back to
    GetActionDataURIs and the source inputs URI goes straight to CreateRun (no upload)."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    mock_client, mock_run_service, mock_dataproxy, _ = _mock_client_with_run()
    mock_dataproxy.get_action_data.side_effect = ConnectError(
        Code.NOT_FOUND, "object 's3://b/metadata/v2/p/d/r1/a0/1/outputs.pb' not found"
    )
    mock_run_service.get_action_data_u_r_is.return_value = run_service_pb2.GetActionDataURIsResponse(
        inputs_uri="s3://b/metadata/v2/p/d/r1/a0/inputs.pb", outputs_uri=""
    )
    await _init_for_testing(client=mock_client, project="test", domain="test")

    with mock.patch("flyte.remote._run.RunDetails") as RD:
        RD.get.aio = AsyncMock(return_value=_fake_prior_run())
        run = await flyte.with_runcontext(mode="remote").rerun.aio("r1", allow_missing_source_outputs=True)

    assert run
    mock_run_service.get_action_data_u_r_is.assert_called_once()
    mock_dataproxy.upload_inputs.assert_not_called()
    req: run_service_pb2.CreateRunRequest = mock_run_service.create_run.call_args[0][0]
    assert req.offloaded_input_data.uri == "s3://b/metadata/v2/p/d/r1/a0/inputs.pb"
    # The server requires a non-empty inputs_hash (min_len=1); a deterministic URI-derived
    # stand-in is sent since the inputs blob can't be read client-side.
    assert len(req.offloaded_input_data.inputs_hash) == 11
    from flyte._run import _uri_inputs_hash

    assert req.offloaded_input_data.inputs_hash == _uri_inputs_hash("s3://b/metadata/v2/p/d/r1/a0/inputs.pb")


@pytest.mark.asyncio
async def test_rerun_missing_source_inputs_stays_fatal():
    """A NotFound that points at the inputs themselves is a real error — no fallback."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    mock_client, mock_run_service, mock_dataproxy, _ = _mock_client_with_run()
    mock_dataproxy.get_action_data.side_effect = ConnectError(
        Code.NOT_FOUND, "object 's3://b/metadata/v2/p/d/r1/a0/1/inputs.pb' not found"
    )
    await _init_for_testing(client=mock_client, project="test", domain="test")

    import flyte.errors

    with mock.patch("flyte.remote._run.RunDetails") as RD:
        RD.get.aio = AsyncMock(return_value=_fake_prior_run())
        with pytest.raises(flyte.errors.RuntimeUserError, match="inputs are no longer in storage"):
            await flyte.with_runcontext(mode="remote").rerun.aio("r1")
    mock_run_service.get_action_data_u_r_is.assert_not_called()


@pytest.mark.asyncio
async def test_rerun_missing_source_outputs_errors_by_default():
    """Without the opt-in, a missing-outputs 404 is a hard error: the client cannot verify
    the inputs blob still exists (the GetActionData 404 is a race between the two halves),
    and silently creating a run with dead inputs strands it at runtime."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    import flyte.errors

    mock_client, mock_run_service, mock_dataproxy, _ = _mock_client_with_run()
    mock_dataproxy.get_action_data.side_effect = ConnectError(
        Code.NOT_FOUND, "object 's3://b/metadata/v2/p/d/r1/a0/1/outputs.pb' not found"
    )
    await _init_for_testing(client=mock_client, project="test", domain="test")

    with mock.patch("flyte.remote._run.RunDetails") as RD:
        RD.get.aio = AsyncMock(return_value=_fake_prior_run())
        with pytest.raises(flyte.errors.RuntimeUserError, match="allow-missing-outputs"):
            await flyte.with_runcontext(mode="remote").rerun.aio("r1")
    mock_run_service.get_action_data_u_r_is.assert_not_called()
    mock_run_service.create_run.assert_not_called()
