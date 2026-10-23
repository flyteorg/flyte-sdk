import pathlib
from unittest.mock import AsyncMock, patch

import pytest
from flyteidl2.common import identifier_pb2, phase_pb2
from flyteidl2.core import interface_pb2, literals_pb2, types_pb2
from flyteidl2.task import common_pb2

import flyte
import flyte.errors
import flyte.report
from flyte._context import internal_ctx
from flyte._internal.controllers.remote._action import Action
from flyte._internal.controllers.remote._controller import RemoteController
from flyte._internal.controllers.remote._input_overrides import ActionInputOverrides, literal_matches_type
from flyte._internal.controllers.remote._service_protocol import ClientSet
from flyte.models import ActionID, CodeBundle, RawDataPath, TaskContext

env = flyte.TaskEnvironment("test-overrides")


@env.task
async def add(x: int, y: int) -> None:
    pass


def _int(v: int) -> literals_pb2.Literal:
    return literals_pb2.Literal(scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(integer=v)))


def _str(v: str) -> literals_pb2.Literal:
    return literals_pb2.Literal(scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(string_value=v)))


def _iface(**types: int) -> interface_pb2.TypedInterface:
    return interface_pb2.TypedInterface(
        inputs=interface_pb2.VariableMap(
            variables=[
                interface_pb2.VariableEntry(key=k, value=interface_pb2.Variable(type=types_pb2.LiteralType(simple=t)))
                for k, t in types.items()
            ]
        )
    )


def _overrides(**by_action: dict) -> ActionInputOverrides:
    o = ActionInputOverrides(uri="s3://bucket/overrides.pb")
    o._overrides = {
        name: common_pb2.Inputs(literals=[common_pb2.NamedLiteral(name=k, value=v) for k, v in values.items()])
        for name, values in by_action.items()
    }
    return o


def _inputs(**values) -> common_pb2.Inputs:
    return common_pb2.Inputs(literals=[common_pb2.NamedLiteral(name=k, value=v) for k, v in values.items()])


@pytest.mark.asyncio
async def test_apply_overlays_by_name_and_keeps_others():
    o = _overrides(child={"y": _int(10)})
    iface = _iface(x=types_pb2.INTEGER, y=types_pb2.INTEGER)
    out = await o.apply("child", "add", _inputs(x=_int(1), y=_int(2)), iface)
    assert [(lit.name, lit.value.scalar.primitive.integer) for lit in out.literals] == [("x", 1), ("y", 10)]
    assert await o.apply("other", "add", _inputs(x=_int(1), y=_int(2)), iface) is None


@pytest.mark.asyncio
async def test_apply_adds_input_missing_from_computed():
    o = _overrides(child={"y": _int(10)})
    out = await o.apply("child", "add", _inputs(x=_int(1)), _iface(x=types_pb2.INTEGER, y=types_pb2.INTEGER))
    assert [lit.name for lit in out.literals] == ["x", "y"]


@pytest.mark.asyncio
async def test_apply_rejects_unknown_input_and_wrong_type():
    iface = _iface(x=types_pb2.INTEGER)
    with pytest.raises(flyte.errors.RuntimeUserError, match="no longer has"):
        await _overrides(child={"z": _int(1)}).apply("child", "add", _inputs(x=_int(1)), iface)
    with pytest.raises(flyte.errors.RuntimeUserError, match="does not match its type"):
        await _overrides(child={"x": _str("a")}).apply("child", "add", _inputs(x=_int(1)), iface)


@pytest.mark.asyncio
async def test_disabled_without_uri(monkeypatch):
    monkeypatch.delenv("_U_ACTION_INPUT_OVERRIDES", raising=False)
    o = ActionInputOverrides()
    assert not o.enabled
    assert await o.apply("child", "add", _inputs(x=_int(1)), _iface(x=types_pb2.INTEGER)) is None


@pytest.mark.asyncio
async def test_unreadable_file_is_a_system_error():
    o = ActionInputOverrides(uri="s3://bucket/missing.pb")
    get_stream = patch("flyte.storage.get_stream", side_effect=FileNotFoundError("nope"))
    with get_stream as stream, patch("asyncio.sleep", new=AsyncMock()):
        with pytest.raises(flyte.errors.RuntimeSystemError, match="action input overrides"):
            await o.apply("child", "add", _inputs(x=_int(1)), _iface(x=types_pb2.INTEGER))
    assert stream.call_count == 3


@pytest.mark.asyncio
@pytest.mark.skipif(
    not hasattr(common_pb2, "ActionInputOverridesData"),
    reason="needs a flyteidl2 build with ActionInputOverridesData",
)
async def test_transient_read_error_is_retried():
    data = common_pb2.ActionInputOverridesData()
    data.overrides["child"].CopyFrom(_inputs(x=_int(7)))

    async def chunks():
        yield data.SerializeToString()

    o = ActionInputOverrides(uri="s3://bucket/overrides.pb")
    side_effect = [ConnectionError("blip"), chunks()]
    with patch("flyte.storage.get_stream", side_effect=side_effect), patch("asyncio.sleep", new=AsyncMock()):
        out = await o.apply("child", "add", _inputs(x=_int(1)), _iface(x=types_pb2.INTEGER))
    assert out.literals[0].value.scalar.primitive.integer == 7


def test_literal_matches_type():
    lt_int = types_pb2.LiteralType(simple=types_pb2.INTEGER)
    assert literal_matches_type(_int(1), lt_int)
    assert not literal_matches_type(_str("a"), lt_int)
    lt_list = types_pb2.LiteralType(collection_type=lt_int)
    coll = literals_pb2.Literal(collection=literals_pb2.LiteralCollection(literals=[_int(1), _int(2)]))
    assert literal_matches_type(coll, lt_list)
    assert not literal_matches_type(_int(1), lt_list)
    lt_union = types_pb2.LiteralType(
        union_type=types_pb2.UnionType(variants=[lt_int, types_pb2.LiteralType(simple=types_pb2.NONE)])
    )
    assert literal_matches_type(_int(3), lt_union)
    assert not literal_matches_type(_str("a"), lt_union)


@pytest.mark.asyncio
async def test_controller_applies_override_and_flags_action():
    async def make_client() -> ClientSet:
        return AsyncMock()  # type: ignore

    done = Action(
        parent_action_name="a0",
        action_id=identifier_pb2.ActionIdentifier(name="child"),
        phase=phase_pb2.ACTION_PHASE_SUCCEEDED,
    )
    with (
        patch(
            "flyte._internal.controllers.remote._controller.upload_inputs_with_retry", new_callable=AsyncMock
        ) as mock_upload,
        patch(
            "flyte._internal.controllers.remote._controller.RemoteController.submit_and_wait_for_action",
            new_callable=AsyncMock,
        ) as mock_submit,
        patch("flyte._initialize.get_init_config") as mock_cfg,
    ):
        mock_cfg.return_value.root_dir = pathlib.Path(__file__).parent
        mock_submit.return_value = done
        ctx = internal_ctx()

        def new_tctx() -> TaskContext:
            return TaskContext(
                action=ActionID(name="a0"),
                raw_data_path=RawDataPath(path="test"),
                output_path="/tmp",
                version="v1",
                run_base_dir="/tmp/outputs/base",
                report=flyte.report.Report(name="r"),
                code_bundle=CodeBundle(
                    computed_version="v", destination=str(pathlib.Path(__file__).parent.absolute()), tgz="d.tgz"
                ),
            )

        # First learn the name the controller computes for add(1, 2), then override that name.
        with ctx.replace_task_context(new_tctx()):
            controller = RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)
            await controller.submit(add, 1, 2)
        name = mock_submit.call_args.args[0].name
        assert mock_submit.call_args.args[0].input_override_applied is False

        # A re-run parent starts with a fresh task context, so the call sequence (and name) repeat.
        with ctx.replace_task_context(new_tctx()):
            controller = RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)
            controller._input_overrides = _overrides(**{name: {"y": _int(40)}})
            await controller.submit(add, 1, 2)

        submitted = mock_submit.call_args.args[0]
        assert submitted.name == name, "the name stays the one computed from the original inputs"
        assert submitted.input_override_applied is True
        uploaded = common_pb2.Inputs()
        uploaded.ParseFromString(mock_upload.call_args.args[0])
        assert {lit.name: lit.value.scalar.primitive.integer for lit in uploaded.literals} == {"x": 1, "y": 40}
