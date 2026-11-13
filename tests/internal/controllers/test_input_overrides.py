import pathlib
from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest
from flyteidl2.common import identifier_pb2, phase_pb2
from flyteidl2.core import literals_pb2
from flyteidl2.task import common_pb2

import flyte
import flyte.errors
import flyte.report
from flyte._context import internal_ctx
from flyte._internal.controllers.remote import _input_overrides
from flyte._internal.controllers.remote._action import Action
from flyte._internal.controllers.remote._controller import RemoteController
from flyte._internal.controllers.remote._input_overrides import child_overrides
from flyte._internal.controllers.remote._service_protocol import ClientSet
from flyte._internal.runtime.convert import ACTION_OVERRIDES_CONTEXT_KEY
from flyte.models import ActionID, CodeBundle, RawDataPath, TaskContext

env = flyte.TaskEnvironment("test-overrides")

URI = "s3://bucket/overrides.pb"


@env.task
async def add(x: int, y: int) -> None:
    pass


@env.task(cache="auto")
async def cached_add(x: int, y: int) -> None:
    pass


def _int(v: int) -> literals_pb2.Literal:
    return literals_pb2.Literal(scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(integer=v)))


def _inputs(**values) -> common_pb2.Inputs:
    return common_pb2.Inputs(literals=[common_pb2.NamedLiteral(name=k, value=v) for k, v in values.items()])


def _ints(inputs: common_pb2.Inputs) -> dict[str, int]:
    return {lit.name: lit.value.scalar.primitive.integer for lit in inputs.literals}


def _context(inputs: common_pb2.Inputs) -> dict[str, str]:
    return {kv.key: kv.value for kv in inputs.context}


def _arg(args, flag: str) -> str:
    return args[list(args).index(flag) + 1]


class _Store:
    """The data plane: serialized protobufs by URI, served through flyte.storage.get_stream."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.reads: list[str] = []

    def put(self, uri: str, message) -> str:
        self.objects[uri] = message.SerializeToString()
        return uri

    def overrides(
        self,
        uri: str = URI,
        inputs: Optional[dict[str, dict]] = None,
        code: Optional[dict[str, str]] = None,
        ancestors: tuple[str, ...] = (),
    ) -> str:
        """Write an overrides file at uri, each action's inputs at their own path; return uri."""
        data = common_pb2.ActionOverridesData(ancestor_actions=ancestors)
        for name, values in (inputs or {}).items():
            data.input_uris[name] = self.put(f"{uri}.{name}.inputs.pb", _inputs(**values))
        for name, tgz in (code or {}).items():
            data.code_bundles[name].tgz = tgz
            data.code_bundles[name].version = "v-patched"
        return self.put(uri, data)

    def get_stream(self, path: str, **_):
        self.reads.append(path)
        if path not in self.objects:
            raise FileNotFoundError(path)

        async def chunks():
            yield self.objects[path]

        return chunks()


@pytest.fixture
def store():
    _input_overrides._overrides_file.cache_clear()
    s = _Store()
    with patch("flyte.storage.get_stream", side_effect=s.get_stream):
        yield s
    _input_overrides._overrides_file.cache_clear()


def _new_tctx(context: Optional[dict] = None, tgz: str = "d.tgz", version: str = "v") -> TaskContext:
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


def _fork(uri: str = URI) -> TaskContext:
    return _new_tctx({ACTION_OVERRIDES_CONTEXT_KEY: uri})


# child_overrides


@pytest.mark.asyncio
async def test_no_overrides_outside_a_fork(store):
    assert await child_overrides(_new_tctx(), "child", "add", _inputs(x=_int(1)), ["x"]) is None
    assert store.reads == []


@pytest.mark.asyncio
async def test_inputs_overlay_by_name_keep_others_and_add_missing(store):
    store.overrides(inputs={"child": {"y": _int(10)}})
    out = await child_overrides(_fork(), "child", "add", _inputs(x=_int(1), y=_int(2)), ["x", "y"])
    assert _ints(out.inputs) == {"x": 1, "y": 10}
    assert out.applied and not out.skip_cache, "an input-overridden action keeps a cache key of its new inputs"
    # An input the computed inputs omitted (e.g. left to its default) still lands.
    out = await child_overrides(_fork(), "child", "add", _inputs(x=_int(1)), ["x", "y"])
    assert [lit.name for lit in out.inputs.literals] == ["x", "y"]
    assert await child_overrides(_fork(), "other", "add", _inputs(x=_int(1)), ["x"]) is None


@pytest.mark.asyncio
async def test_override_of_an_input_the_task_no_longer_has_is_a_user_error(store):
    store.overrides(inputs={"child": {"z": _int(1)}})
    with pytest.raises(flyte.errors.RuntimeUserError, match="no longer has"):
        await child_overrides(_fork(), "child", "add", _inputs(x=_int(1)), ["x"])


@pytest.mark.asyncio
async def test_file_is_read_once_and_inputs_only_for_the_actions_enqueued(store):
    store.overrides(inputs={"child": {"x": _int(7)}, "other": {"x": _int(8)}})
    for _ in range(2):
        await child_overrides(_fork(), "child", "add", _inputs(x=_int(1)), ["x"])
    await child_overrides(_fork(), "unrelated", "add", _inputs(x=_int(1)), ["x"])
    assert store.reads == [URI, f"{URI}.child.inputs.pb", f"{URI}.child.inputs.pb"]


@pytest.mark.asyncio
async def test_unreadable_file_or_inputs_are_system_errors(store):
    with pytest.raises(flyte.errors.RuntimeSystemError, match="action overrides of this run"):
        await child_overrides(_fork("s3://bucket/missing.pb"), "child", "add", _inputs(x=_int(1)), ["x"])
    store.overrides()
    data = common_pb2.ActionOverridesData()
    data.input_uris["child"] = "s3://bucket/gone.pb"
    store.put(URI, data)
    with pytest.raises(flyte.errors.RuntimeSystemError, match="override inputs of action child"):
        await child_overrides(_fork(), "child", "add", _inputs(x=_int(1)), ["x"])


@pytest.mark.asyncio
async def test_code_bundle_replaces_tgz_keeps_destination_and_skips_cache(store):
    store.overrides(code={"child": "s3://bucket/fastpatched.tar.gz"})
    parent = CodeBundle(computed_version="v-src", destination="/root", tgz="s3://bucket/fastsrc.tar.gz")
    out = await child_overrides(_fork(), "child", "add", _inputs(x=_int(1)), ["x"], parent)
    assert (out.code_bundle.tgz, out.code_bundle.destination, out.code_bundle.computed_version) == (
        "s3://bucket/fastpatched.tar.gz",
        "/root",
        "v-patched",
    )
    assert out.inputs is None, "a code-only override changes no inputs"
    assert out.applied and out.skip_cache


@pytest.mark.asyncio
async def test_code_bundle_needs_a_source_bundle_and_a_non_deployed_task(store):
    store.overrides(code={"child": "s3://bucket/fastpatched.tar.gz"})
    for parent in (None, CodeBundle(computed_version="v", pkl="s3://bucket/code.pkl.gz")):
        with pytest.raises(flyte.errors.RuntimeUserError, match="no bundle to replace"):
            await child_overrides(_fork(), "child", "add", _inputs(x=_int(1)), ["x"], parent)
    with pytest.raises(flyte.errors.RuntimeUserError, match="fixed by its registration"):
        await child_overrides(_fork(), "child", "add", _inputs(x=_int(1)), ["x"], task_ref=True)


@pytest.mark.asyncio
async def test_ancestor_skips_the_cache_without_applying_anything(store):
    store.overrides(ancestors=("parent",))
    out = await child_overrides(_fork(), "parent", "add", _inputs(x=_int(1)), ["x"])
    assert out.skip_cache and not out.applied and out.inputs is None
    assert await child_overrides(_fork(), "child", "add", _inputs(x=_int(1)), ["x"]) is None


def test_files_are_cached_by_uri():
    _input_overrides._overrides_file.cache_clear()
    a = _input_overrides._overrides_file("s3://bucket/a.pb")
    assert _input_overrides._overrides_file("s3://bucket/a.pb") is a
    assert _input_overrides._overrides_file("s3://bucket/b.pb") is not a


# The remote controller


class _Harness:
    """A RemoteController whose uploads and enqueues are captured instead of sent."""

    def __init__(self):
        self.uploads: list[common_pb2.Inputs] = []
        self.submitted: list[Action] = []

    async def submit(self, tctx: TaskContext, task, *args, controller: Optional[RemoteController] = None):
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


@pytest.mark.asyncio
async def test_controller_applies_override_and_flags_action(store):
    h = _Harness()
    # First learn the name the controller computes for add(1, 2), then override that name. A
    # re-run parent starts with a fresh task context, so the call sequence (and name) repeat.
    await h.submit(_new_tctx(), add, 1, 2)
    name = h.submitted[-1].name
    assert h.submitted[-1].override_applied is False

    store.overrides(inputs={name: {"y": _int(40)}})
    await h.submit(_fork(), add, 1, 2)
    assert h.submitted[-1].name == name, "the name stays the one computed from the original inputs"
    assert h.submitted[-1].override_applied is True
    assert _ints(h.uploads[-1]) == {"x": 1, "y": 40}


@pytest.mark.asyncio
async def test_controller_runs_code_override_from_its_bundle_without_cache(store):
    h = _Harness()
    src = {"tgz": "s3://bucket/fastsrc.tar.gz", "version": "v-src"}
    await h.submit(_new_tctx(**src), cached_add, 1, 2)
    original = h.submitted[-1]
    assert original.cache_key, "the task is cached without an override"
    assert _arg(original.task.task_template.container.args, "--tgz") == "s3://bucket/fastsrc.tar.gz"

    store.overrides(code={original.name: "s3://bucket/fastpatched.tar.gz"})
    await h.submit(_new_tctx({ACTION_OVERRIDES_CONTEXT_KEY: URI}, **src), cached_add, 1, 2)
    submitted = h.submitted[-1]
    assert submitted.name == original.name, "the name stays the one computed from the parent's code"
    assert submitted.override_applied is True
    assert submitted.cache_key is None, "a cache hit would return the original code's result"
    args = submitted.task.task_template.container.args
    assert _arg(args, "--tgz") == "s3://bucket/fastpatched.tar.gz"
    assert _arg(args, "--dest") == _arg(original.task.task_template.container.args, "--dest")


@pytest.mark.asyncio
async def test_ancestor_is_enqueued_without_a_cache_key(store):
    h = _Harness()
    await h.submit(_new_tctx(), cached_add, 1, 2)
    original = h.submitted[-1]
    assert original.cache_key

    store.overrides(ancestors=(original.name,))
    await h.submit(_fork(), cached_add, 1, 2)
    ancestor = h.submitted[-1]
    assert ancestor.name == original.name
    assert ancestor.cache_key is None, "a cache hit would skip the override beneath it"
    assert ancestor.override_applied is False


@pytest.mark.asyncio
async def test_two_runs_through_one_controller_apply_their_own_overrides(store):
    h = _Harness()
    controller = await h.submit(_new_tctx(), add, 1, 2)
    name = h.submitted[-1].name
    store.overrides("s3://bucket/run-a.pb", inputs={name: {"y": _int(10)}})
    store.overrides("s3://bucket/run-b.pb", inputs={name: {"y": _int(20)}})

    for uri, y in (("s3://bucket/run-a.pb", 10), ("s3://bucket/run-b.pb", 20), (None, 2)):
        await h.submit(_fork(uri) if uri else _new_tctx(), add, 1, 2, controller=controller)
        assert _ints(h.uploads[-1]) == {"x": 1, "y": y}
        assert h.submitted[-1].override_applied is (uri is not None)


@pytest.mark.asyncio
async def test_key_reaches_a_grandchild_through_a_custom_context_block(store):
    from flyte._internal.runtime.convert import Inputs

    store.overrides()  # no override for these actions: only the key's propagation matters
    h = _Harness()
    tctx = _new_tctx({ACTION_OVERRIDES_CONTEXT_KEY: URI, "team": "ml"})

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
    with internal_ctx().replace_task_context(_fork()):
        with pytest.raises(ValueError, match="reserved for internal use"):
            with flyte.custom_context(**{ACTION_OVERRIDES_CONTEXT_KEY: "s3://elsewhere.pb"}):
                pass
        assert flyte.get_custom_context() == {}
        assert internal_ctx().data.task_context.custom_context[ACTION_OVERRIDES_CONTEXT_KEY] == URI


def test_rust_controller_fails_an_override_it_cannot_record():
    pytest.importorskip("flyte_controller_base")
    from flyte._internal.controllers.remote._input_overrides import ChildOverrides
    from flyte._internal.controllers.remote._r_controller import _reject_applied_overrides

    _reject_applied_overrides(None, "child")
    # An ancestor only skips its cache key, which the Rust controller applies.
    _reject_applied_overrides(ChildOverrides(inputs=None, code_bundle=None, skip_cache=True), "parent")
    with pytest.raises(flyte.errors.RuntimeSystemError, match="Rust controller"):
        _reject_applied_overrides(ChildOverrides(inputs=_inputs(x=_int(1)), code_bundle=None, skip_cache=False), "c")
