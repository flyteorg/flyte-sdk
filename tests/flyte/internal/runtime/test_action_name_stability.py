"""
Action-name stability across runs (ENG26-1042).

Recovery matches completed actions from a previous run by action name, so the name must fold
in only the task's position, inputs, and per-task code identity — never the code-bundle
version, container image, or other spec fields that change on every code edit or deploy.
"""

from __future__ import annotations

import pathlib

from flyteidl2.core import identifier_pb2, interface_pb2, literals_pb2, tasks_pb2, types_pb2
from flyteidl2.task import common_pb2

import flyte
from flyte._internal.runtime import convert
from flyte._internal.runtime.task_serde import translate_task_to_wire
from flyte.models import ActionID, RawDataPath, SerializationContext, TaskContext
from flyte.report import Report


def _make_template(
    name: str = "env.my_task",
    image: str = "img:v1",
    args: tuple[str, ...] = ("--version", "v1"),
    discovery_version: str = "body-hash-1",
    input_type: int = types_pb2.INTEGER,
) -> tasks_pb2.TaskTemplate:
    return tasks_pb2.TaskTemplate(
        id=identifier_pb2.Identifier(name=name, version="run-version"),
        type="python",
        metadata=tasks_pb2.TaskMetadata(discovery_version=discovery_version),
        interface=interface_pb2.TypedInterface(
            inputs=interface_pb2.VariableMap(
                variables=[
                    interface_pb2.VariableEntry(
                        key="x",
                        value=interface_pb2.Variable(type=types_pb2.LiteralType(simple=input_type)),
                    )
                ]
            )
        ),
        container=tasks_pb2.Container(image=image, args=list(args)),
    )


def _make_tctx() -> TaskContext:
    return TaskContext(
        action=ActionID(name="parent", run_name="run1", project="p", domain="d"),
        run_base_dir="s3://bucket/metadata/p/d/run1",
        version="v1",
        raw_data_path=RawDataPath(path="s3://bucket/raw/p/d/run1"),
        output_path="s3://bucket/output/p/d/run1",
        report=Report(name="test"),
    )


def _named_literal(name: str, value: int) -> common_pb2.NamedLiteral:
    return common_pb2.NamedLiteral(
        name=name,
        value=literals_pb2.Literal(scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(integer=value))),
    )


class TestTaskIdentityHash:
    def test_stable_across_code_bundle_and_image_changes(self):
        """A new code bundle / image (any code edit or redeploy) must not change the identity."""
        t1 = _make_template(image="img:v1", args=("--version", "v1"))
        t2 = _make_template(image="img:v2", args=("--version", "v2"))
        assert convert.generate_task_identity_hash(t1) == convert.generate_task_identity_hash(t2)

    def test_stable_across_resource_and_env_changes(self):
        t1 = _make_template()
        t2 = _make_template()
        t2.container.resources.requests.add(name=tasks_pb2.Resources.CPU, value="2")
        t2.container.env.add(key="FOO", value="bar")
        assert convert.generate_task_identity_hash(t1) == convert.generate_task_identity_hash(t2)

    def test_changes_with_discovery_version(self):
        """Editing the task's own function body changes discovery_version → new identity."""
        t1 = _make_template(discovery_version="body-hash-1")
        t2 = _make_template(discovery_version="body-hash-2")
        assert convert.generate_task_identity_hash(t1) != convert.generate_task_identity_hash(t2)

    def test_changes_with_task_name(self):
        t1 = _make_template(name="env.task_a")
        t2 = _make_template(name="env.task_b")
        assert convert.generate_task_identity_hash(t1) != convert.generate_task_identity_hash(t2)

    def test_changes_with_interface(self):
        t1 = _make_template(input_type=types_pb2.INTEGER)
        t2 = _make_template(input_type=types_pb2.STRING)
        assert convert.generate_task_identity_hash(t1) != convert.generate_task_identity_hash(t2)

    def test_action_name_stable_across_code_bundle_changes(self):
        """End to end: same position + inputs + code identity → same action name."""
        tctx = _make_tctx()
        inputs_hash = "abc123"
        id1, _ = convert.generate_sub_action_id_and_output_path(
            tctx, convert.generate_task_identity_hash(_make_template(image="img:v1")), inputs_hash, 1
        )
        id2, _ = convert.generate_sub_action_id_and_output_path(
            tctx, convert.generate_task_identity_hash(_make_template(image="img:v2")), inputs_hash, 1
        )
        assert id1.name == id2.name


class TestFilteredInputsHash:
    def test_ignored_vars_excluded(self):
        """Two input sets differing only in an ignored var must hash identically."""
        inputs_a = common_pb2.Inputs(literals=[_named_literal("x", 1), _named_literal("ts", 100)])
        inputs_b = common_pb2.Inputs(literals=[_named_literal("x", 1), _named_literal("ts", 200)])
        assert convert.generate_filtered_inputs_hash(inputs_a, ["ts"]) == convert.generate_filtered_inputs_hash(
            inputs_b, ["ts"]
        )

    def test_non_ignored_vars_still_distinguish(self):
        inputs_a = common_pb2.Inputs(literals=[_named_literal("x", 1), _named_literal("ts", 100)])
        inputs_b = common_pb2.Inputs(literals=[_named_literal("x", 2), _named_literal("ts", 100)])
        assert convert.generate_filtered_inputs_hash(inputs_a, ["ts"]) != convert.generate_filtered_inputs_hash(
            inputs_b, ["ts"]
        )


class TestTraceActionIdentity:
    def test_deterministic(self):
        def my_trace(x: int) -> int:
            return x + 1

        assert convert.generate_trace_action_identity(my_trace) == convert.generate_trace_action_identity(my_trace)

    def test_changes_with_function_body(self):
        def my_trace(x: int) -> int:
            return x + 1

        def my_trace_edited(x: int) -> int:
            return x + 2

        id_orig = convert.generate_trace_action_identity(my_trace)
        id_edited = convert.generate_trace_action_identity(my_trace_edited)
        assert id_orig != id_edited

    def test_falls_back_to_name_without_source(self):
        ns: dict = {}
        exec("def no_source(x):\n    return x\n", ns)
        assert convert.generate_trace_action_identity(ns["no_source"]) == "no_source"


class TestDiscoveryVersionAlwaysPopulated:
    def test_cache_disabled_still_gets_version(self):
        """The spec must carry a per-task code version even with caching disabled, since
        deterministic action names depend on it."""
        env = flyte.TaskEnvironment(name="stability_env")

        @env.task(cache="disable")
        async def t_disabled(a: int) -> int:
            return a

        @env.task(cache=flyte.Cache(behavior="auto"))
        async def t_auto(a: int) -> int:
            return a

        sctx = SerializationContext(
            project="p",
            domain="d",
            version="v-test",
            org="o",
            input_path="/tmp/inputs",
            output_path="/tmp/outputs",
            image_cache=None,
            code_bundle=None,
            root_dir=pathlib.Path.cwd(),
        )

        spec_disabled = translate_task_to_wire(t_disabled, sctx)
        spec_auto = translate_task_to_wire(t_auto, sctx)

        assert spec_disabled.task_template.metadata.discoverable is False
        assert spec_disabled.task_template.metadata.discovery_version != ""
        assert spec_auto.task_template.metadata.discoverable is True
        assert spec_auto.task_template.metadata.discovery_version != ""
