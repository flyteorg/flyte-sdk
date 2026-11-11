import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from flyteidl2.common import identifier_pb2, phase_pb2
from flyteidl2.core import interface_pb2, literals_pb2, types_pb2
from flyteidl2.task import common_pb2

import flyte
import flyte.errors
import flyte.report
from flyte._constants import ACTION_OVERRIDES_CONTEXT_KEY
from flyte._context import internal_ctx
from flyte._internal.controllers.remote import _input_overrides
from flyte._internal.controllers.remote._action import Action
from flyte._internal.controllers.remote._controller import RemoteController
from flyte._internal.controllers.remote._input_overrides import ActionOverrides, literal_matches_type, overrides_for
from flyte._internal.controllers.remote._service_protocol import ClientSet
from flyte.models import ActionID, CodeBundle, RawDataPath, TaskContext

env = flyte.TaskEnvironment("test-overrides")

URI = "s3://bucket/overrides.pb"


@pytest.fixture(autouse=True)
def _fresh_overrides_cache():
    _input_overrides._by_uri.clear()
    yield
    _input_overrides._by_uri.clear()


def _install(o: ActionOverrides, uri: str = URI) -> ActionOverrides:
    """Make o the loaded overrides of the run whose actions carry uri in their context."""
    _input_overrides._by_uri[uri] = o
    return o


def _new_tctx(context: dict | None = None, tgz: str = "d.tgz", version: str = "v") -> TaskContext:
    return TaskContext(
        action=ActionID(name="a0"),
        raw_data_path=RawDataPath(path="test"),
        output_path="/tmp",
        version="v1",
        run_base_dir="/tmp/outputs/base",
        report=flyte.report.Report(name="r"),
        code_bundle=CodeBundle(
            computed_version=version, destination=str(pathlib.Path(__file__).parent.absolute()), tgz=tgz
        ),
        custom_context=dict(context or {}),
    )


@env.task
async def add(x: int, y: int) -> None:
    pass


@env.task(cache="auto")
async def cached_add(x: int, y: int) -> None:
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


def _overrides(uri: str = URI, **by_action: dict) -> ActionOverrides:
    """Overrides already loaded from the file, with each action's inputs served from its own path."""
    o = ActionOverrides(uri=uri)
    o._input_uris = {name: f"s3://bucket/{name}/inputs.pb" for name in by_action}
    stored = {
        f"s3://bucket/{name}/inputs.pb": common_pb2.Inputs(
            literals=[common_pb2.NamedLiteral(name=k, value=v) for k, v in values.items()]
        )
        for name, values in by_action.items()
    }

    async def read(uri: str, what: str, new_message):
        message = new_message()
        message.CopyFrom(stored[uri])
        return message

    o._read = read  # type: ignore[method-assign]
    return o


def _stream(by_uri: dict):
    """A storage.get_stream stand-in serving serialized protobufs by URI."""

    def get_stream(uri: str):
        async def chunks():
            yield by_uri[uri].SerializeToString()

        return chunks()

    return get_stream


def _with_code(o: ActionOverrides, **by_action: str) -> ActionOverrides:
    """Give actions a replacement code bundle (an ActionCodeBundle stand-in: the pinned flyteidl2 predates it)."""
    o._code_bundles = {name: SimpleNamespace(tgz=tgz, version="v-patched") for name, tgz in by_action.items()}
    return o


def _arg(args, flag: str) -> str:
    return args[list(args).index(flag) + 1]


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
async def test_disabled_without_uri():
    o = ActionOverrides()
    assert not o.enabled
    assert await o.apply("child", "add", _inputs(x=_int(1)), _iface(x=types_pb2.INTEGER)) is None


@pytest.mark.asyncio
async def test_flyteidl2_without_overrides_fails_without_reading(monkeypatch):
    monkeypatch.delattr(common_pb2, "ActionOverridesData", raising=False)
    o = ActionOverrides(uri="s3://bucket/overrides.pb")
    with patch("flyte.storage.get_stream") as stream:
        with pytest.raises(flyte.errors.NonRecoverableError, match="predates"):
            await o.apply("child", "add", _inputs(x=_int(1)), _iface(x=types_pb2.INTEGER))
    stream.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not hasattr(common_pb2, "ActionOverridesData"),
    reason="needs a flyteidl2 build with ActionOverridesData",
)
async def test_unreadable_file_is_a_system_error():
    o = ActionOverrides(uri="s3://bucket/missing.pb")
    get_stream = patch("flyte.storage.get_stream", side_effect=FileNotFoundError("nope"))
    with get_stream as stream, patch("asyncio.sleep", new=AsyncMock()):
        with pytest.raises(flyte.errors.RuntimeSystemError, match="action input overrides"):
            await o.apply("child", "add", _inputs(x=_int(1)), _iface(x=types_pb2.INTEGER))
    assert stream.call_count == 3


@pytest.mark.asyncio
@pytest.mark.skipif(
    not hasattr(common_pb2, "ActionOverridesData"),
    reason="needs a flyteidl2 build with ActionOverridesData",
)
async def test_transient_read_error_is_retried():
    data = common_pb2.ActionOverridesData()
    data.input_uris["child"] = "s3://bucket/child/inputs.pb"
    serve = _stream({"s3://bucket/overrides.pb": data, "s3://bucket/child/inputs.pb": _inputs(x=_int(7))})
    blips = {"s3://bucket/overrides.pb": 1, "s3://bucket/child/inputs.pb": 1}

    def get_stream(uri: str):
        if blips[uri]:
            blips[uri] -= 1
            raise ConnectionError("blip")
        return serve(uri)

    o = ActionOverrides(uri="s3://bucket/overrides.pb")
    with patch("flyte.storage.get_stream", side_effect=get_stream), patch("asyncio.sleep", new=AsyncMock()):
        out = await o.apply("child", "add", _inputs(x=_int(1)), _iface(x=types_pb2.INTEGER))
    assert out.literals[0].value.scalar.primitive.integer == 7


@pytest.mark.asyncio
@pytest.mark.skipif(
    not hasattr(common_pb2, "ActionOverridesData"),
    reason="needs a flyteidl2 build with ActionOverridesData",
)
async def test_inputs_are_read_from_each_actions_path_only_when_it_is_enqueued():
    data = common_pb2.ActionOverridesData()
    data.input_uris["child"] = "s3://bucket/child/inputs.pb"
    data.input_uris["other"] = "s3://bucket/other/inputs.pb"
    serve = _stream({"s3://bucket/overrides.pb": data, "s3://bucket/child/inputs.pb": _inputs(x=_int(7))})

    o = ActionOverrides(uri="s3://bucket/overrides.pb")
    with patch("flyte.storage.get_stream", side_effect=serve) as stream:
        out = await o.apply("child", "add", _inputs(x=_int(1)), _iface(x=types_pb2.INTEGER))
        assert await o.apply("unrelated", "add", _inputs(x=_int(1)), _iface(x=types_pb2.INTEGER)) is None
    assert out.literals[0].value.scalar.primitive.integer == 7
    # The file once, then only the inputs of the action this pod enqueued; never "other"'s.
    assert [c.args[0] for c in stream.call_args_list] == ["s3://bucket/overrides.pb", "s3://bucket/child/inputs.pb"]


@pytest.mark.asyncio
async def test_unreadable_action_inputs_are_a_system_error():
    o = _overrides(child={"x": _int(7)})
    del o._read  # use the real reader
    with patch("flyte.storage.get_stream", side_effect=FileNotFoundError("nope")) as stream:
        with patch("asyncio.sleep", new=AsyncMock()):
            with pytest.raises(flyte.errors.RuntimeSystemError, match="override inputs of action child"):
                await o.apply("child", "add", _inputs(x=_int(1)), _iface(x=types_pb2.INTEGER))
    assert stream.call_count == 3


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

        def new_tctx(context: dict | None = None) -> TaskContext:
            return _new_tctx(context)

        # First learn the name the controller computes for add(1, 2), then override that name.
        with ctx.replace_task_context(new_tctx()):
            controller = RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)
            await controller.submit(add, 1, 2)
        name = mock_submit.call_args.args[0].name
        assert mock_submit.call_args.args[0].override_applied is False

        # A re-run parent starts with a fresh task context, so the call sequence (and name) repeat.
        _install(_overrides(**{name: {"y": _int(40)}}))
        with ctx.replace_task_context(new_tctx({ACTION_OVERRIDES_CONTEXT_KEY: URI})):
            controller = RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)
            await controller.submit(add, 1, 2)

        submitted = mock_submit.call_args.args[0]
        assert submitted.name == name, "the name stays the one computed from the original inputs"
        assert submitted.override_applied is True
        uploaded = common_pb2.Inputs()
        uploaded.ParseFromString(mock_upload.call_args.args[0])
        assert {lit.name: lit.value.scalar.primitive.integer for lit in uploaded.literals} == {"x": 1, "y": 40}


@pytest.mark.asyncio
async def test_code_bundle_replaces_tgz_and_keeps_destination():
    o = _with_code(_overrides(), child="s3://bucket/fastpatched.tar.gz")
    parent = CodeBundle(computed_version="v-src", destination="/root", tgz="s3://bucket/fastsrc.tar.gz")
    bundle = await o.code_bundle("child", "add", parent)
    assert (bundle.tgz, bundle.destination, bundle.computed_version) == (
        "s3://bucket/fastpatched.tar.gz",
        "/root",
        "v-patched",
    )
    assert await o.code_bundle("other", "add", parent) is None
    assert await o.has_code_bundle("child") and not await o.has_code_bundle("other")


@pytest.mark.asyncio
async def test_code_bundle_needs_a_source_bundle_to_replace():
    o = _with_code(_overrides(), child="s3://bucket/fastpatched.tar.gz")
    with pytest.raises(flyte.errors.RuntimeUserError, match="no bundle to replace"):
        await o.code_bundle("child", "add", None)
    pkl = CodeBundle(computed_version="v", pkl="s3://bucket/code.pkl.gz")
    with pytest.raises(flyte.errors.RuntimeUserError, match="no bundle to replace"):
        await o.code_bundle("child", "add", pkl)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not hasattr(common_pb2, "ActionCodeBundle"),
    reason="needs a flyteidl2 build with ActionCodeBundle",
)
async def test_code_bundles_are_read_from_the_file():
    data = common_pb2.ActionOverridesData()
    data.code_bundles["child"].tgz = "s3://bucket/fastpatched.tar.gz"
    data.code_bundles["child"].version = "abc"

    async def chunks():
        yield data.SerializeToString()

    o = ActionOverrides(uri="s3://bucket/overrides.pb")
    parent = CodeBundle(computed_version="v", destination=".", tgz="s3://bucket/fastsrc.tar.gz")
    with patch("flyte.storage.get_stream", return_value=chunks()):
        bundle = await o.code_bundle("child", "add", parent)
    assert (bundle.tgz, bundle.computed_version) == ("s3://bucket/fastpatched.tar.gz", "abc")
    # A code-only override changes no inputs.
    assert await o.apply("child", "add", _inputs(x=_int(1)), _iface(x=types_pb2.INTEGER)) is None


@pytest.mark.asyncio
async def test_controller_runs_code_override_from_its_bundle_without_cache():
    async def make_client() -> ClientSet:
        return AsyncMock()  # type: ignore

    done = Action(
        parent_action_name="a0",
        action_id=identifier_pb2.ActionIdentifier(name="child"),
        phase=phase_pb2.ACTION_PHASE_SUCCEEDED,
    )
    with (
        patch("flyte._internal.controllers.remote._controller.upload_inputs_with_retry", new_callable=AsyncMock),
        patch(
            "flyte._internal.controllers.remote._controller.RemoteController.submit_and_wait_for_action",
            new_callable=AsyncMock,
        ) as mock_submit,
        patch("flyte._initialize.get_init_config") as mock_cfg,
    ):
        mock_cfg.return_value.root_dir = pathlib.Path(__file__).parent
        mock_submit.return_value = done
        ctx = internal_ctx()

        def new_tctx(context: dict | None = None) -> TaskContext:
            return _new_tctx(context, tgz="s3://bucket/fastsrc.tar.gz", version="v-src")

        with ctx.replace_task_context(new_tctx()):
            controller = RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)
            await controller.submit(cached_add, 1, 2)
        original = mock_submit.call_args.args[0]
        assert original.cache_key, "the task is cached without an override"
        assert _arg(original.task.task_template.container.args, "--tgz") == "s3://bucket/fastsrc.tar.gz"

        _install(_with_code(_overrides(), **{original.name: "s3://bucket/fastpatched.tar.gz"}))
        with ctx.replace_task_context(new_tctx({ACTION_OVERRIDES_CONTEXT_KEY: URI})):
            controller = RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)
            await controller.submit(cached_add, 1, 2)

        submitted = mock_submit.call_args.args[0]
        assert submitted.name == original.name, "the name stays the one computed from the parent's code"
        assert submitted.override_applied is True
        assert submitted.cache_key is None, "a cache hit would return the original code's result"
        args = submitted.task.task_template.container.args
        assert _arg(args, "--tgz") == "s3://bucket/fastpatched.tar.gz"
        assert _arg(args, "--dest") == _arg(original.task.task_template.container.args, "--dest")


class _Harness:
    """A RemoteController whose uploads and enqueues are captured instead of sent."""

    def __init__(self):
        self.uploads: list[common_pb2.Inputs] = []
        self.submitted: list[Action] = []

    async def submit(self, tctx: TaskContext, task, *args, controller: RemoteController | None = None):
        async def make_client() -> ClientSet:
            return AsyncMock()  # type: ignore

        async def upload(data: bytes, uri: str, max_bytes=None):
            inputs = common_pb2.Inputs()
            inputs.ParseFromString(data)
            self.uploads.append(inputs)

        async def submit_and_wait(_self, action: Action):
            self.submitted.append(action)
            return Action(
                parent_action_name="a0",
                action_id=identifier_pb2.ActionIdentifier(name=action.name),
                phase=phase_pb2.ACTION_PHASE_SUCCEEDED,
            )

        with (
            patch("flyte._internal.controllers.remote._controller.upload_inputs_with_retry", new=upload),
            patch(
                "flyte._internal.controllers.remote._controller.RemoteController.submit_and_wait_for_action",
                new=submit_and_wait,
            ),
            patch("flyte._initialize.get_init_config") as mock_cfg,
        ):
            mock_cfg.return_value.root_dir = pathlib.Path(__file__).parent
            with internal_ctx().replace_task_context(tctx):
                controller = controller or RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)
                await controller.submit(task, *args)
        return controller


def _context(inputs: common_pb2.Inputs) -> dict[str, str]:
    return {kv.key: kv.value for kv in inputs.context}


def test_overrides_are_cached_by_uri():
    assert overrides_for(None) is None and overrides_for("") is None
    a = overrides_for("s3://bucket/a.pb")
    assert overrides_for("s3://bucket/a.pb") is a
    assert overrides_for("s3://bucket/b.pb") is not a


@pytest.mark.asyncio
async def test_key_reaches_a_grandchild_through_a_custom_context_block():
    from flyte._internal.runtime.convert import Inputs

    h = _Harness()
    tctx = _new_tctx({ACTION_OVERRIDES_CONTEXT_KEY: URI, "team": "ml"})
    _install(_overrides())  # no override for these actions: only the key's propagation matters

    async def in_block():
        with internal_ctx().replace_task_context(tctx):
            with flyte.custom_context(extra="1"):
                assert flyte.get_custom_context() == {"team": "ml", "extra": "1"}
                return internal_ctx().data.task_context

    await h.submit(await in_block(), add, 1, 2)
    child_inputs = h.uploads[-1]
    assert _context(child_inputs) == {ACTION_OVERRIDES_CONTEXT_KEY: URI, "team": "ml", "extra": "1"}

    # The child's task context is seeded from its Inputs.context (as taskrunner does), so the key
    # reaches the actions it launches in turn.
    child_context = Inputs(proto_inputs=child_inputs).context
    assert child_context[ACTION_OVERRIDES_CONTEXT_KEY] == URI
    await h.submit(_new_tctx(child_context), add, 3, 4)
    assert _context(h.uploads[-1])[ACTION_OVERRIDES_CONTEXT_KEY] == URI


@pytest.mark.asyncio
async def test_custom_context_cannot_set_the_key():
    with internal_ctx().replace_task_context(_new_tctx({ACTION_OVERRIDES_CONTEXT_KEY: URI})):
        with pytest.raises(ValueError, match="reserved for internal use"):
            with flyte.custom_context(**{ACTION_OVERRIDES_CONTEXT_KEY: "s3://elsewhere.pb"}):
                pass
        assert flyte.get_custom_context() == {}
        assert internal_ctx().data.task_context.custom_context[ACTION_OVERRIDES_CONTEXT_KEY] == URI


@pytest.mark.asyncio
async def test_two_runs_through_one_controller_apply_their_own_overrides():
    h = _Harness()
    controller = await h.submit(_new_tctx(), add, 1, 2)
    name = h.submitted[-1].name
    _install(_overrides(uri="s3://bucket/run-a.pb", **{name: {"y": _int(10)}}), "s3://bucket/run-a.pb")
    _install(_overrides(uri="s3://bucket/run-b.pb", **{name: {"y": _int(20)}}), "s3://bucket/run-b.pb")

    for uri, y in (("s3://bucket/run-a.pb", 10), ("s3://bucket/run-b.pb", 20), (None, 2)):
        context = {ACTION_OVERRIDES_CONTEXT_KEY: uri} if uri else {}
        await h.submit(_new_tctx(context), add, 1, 2, controller=controller)
        assert {lit.name: lit.value.scalar.primitive.integer for lit in h.uploads[-1].literals} == {"x": 1, "y": y}
        assert h.submitted[-1].override_applied is (uri is not None)


@pytest.mark.asyncio
async def test_ancestor_is_enqueued_without_a_cache_key():
    h = _Harness()
    await h.submit(_new_tctx(), cached_add, 1, 2)
    original = h.submitted[-1]
    assert original.cache_key

    o = _overrides()
    o._ancestors = frozenset({original.name})
    _install(o)
    await h.submit(_new_tctx({ACTION_OVERRIDES_CONTEXT_KEY: URI}), cached_add, 1, 2)
    ancestor = h.submitted[-1]
    assert ancestor.name == original.name
    assert ancestor.cache_key is None, "a cache hit would skip the override beneath it"
    assert ancestor.override_applied is False


@pytest.mark.asyncio
@pytest.mark.skipif(
    not hasattr(common_pb2, "ActionOverridesData")
    or "ancestor_actions" not in common_pb2.ActionOverridesData.DESCRIPTOR.fields_by_name,
    reason="needs a flyteidl2 build with ActionOverridesData.ancestor_actions",
)
async def test_ancestors_are_read_from_the_file():
    data = common_pb2.ActionOverridesData()
    data.ancestor_actions.extend(["parent"])

    async def chunks():
        yield data.SerializeToString()

    o = ActionOverrides(uri=URI)
    with patch("flyte.storage.get_stream", return_value=chunks()):
        assert await o.is_ancestor("parent")
    assert not await o.is_ancestor("child")
