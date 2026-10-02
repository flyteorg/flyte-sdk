"""Tests for `flyte.sandbox.orchestrator`, whose source runs on a sandbox leaseworker."""

import asyncio
from types import SimpleNamespace

import pytest
from flyteidl2.task import task_definition_pb2
from google.protobuf.json_format import MessageToDict

import flyte
import flyte.errors
import flyte.sandbox
from flyte._internal.runtime.task_serde import translate_task_to_wire
from flyte.models import SerializationContext
from flyte.remote._task import LazyEntity
from flyte.sandbox import OrchestratorTaskTemplate


def _remote(name: str, version: str = "v7") -> LazyEntity:
    """A remote task reference that resolves without a backend."""

    async def getter():
        task_id = task_definition_pb2.TaskIdentifier(
            org="org", project="proj", domain="dev", name=name, version=version
        )
        return SimpleNamespace(pb2=SimpleNamespace(task_id=task_id))

    return LazyEntity(name, getter)


add = _remote("math.add")
multiply = _remote("math.multiply")

env = flyte.TaskEnvironment(name="local-env")


@env.task
async def local_double(x: int) -> int:
    return x * 2


@flyte.sandbox.orchestrator(queue="sandbox", child_queue="default")
def pipeline(x: int, y: int) -> int:
    total = add(x, y)
    return multiply(total, 2)


def _serialize(task: OrchestratorTaskTemplate):
    asyncio.run(task.resolve_tasks())
    return translate_task_to_wire(task, SerializationContext(version="v1")).task_template


class TestSerialization:
    def test_is_shipped_without_a_container(self):
        template = _serialize(pipeline)
        assert template.type == "sandbox-orchestrator"
        assert template.task_type_version == 1
        assert not template.HasField("container")
        assert not template.HasField("k8s_pod")

    def test_template_carries_source_inputs_and_pinned_tasks(self):
        custom = MessageToDict(_serialize(pipeline).custom)
        assert custom["input_names"] == ["x", "y"]
        assert custom["source"].startswith("def pipeline(x: int, y: int) -> int:")
        assert custom["source"].rstrip().endswith("pipeline(x, y)")
        assert "@flyte.sandbox" not in custom["source"]
        assert custom["tasks"] == {
            "add": {"project": "proj", "domain": "dev", "name": "math.add", "version": "v7"},
            "multiply": {"project": "proj", "domain": "dev", "name": "math.multiply", "version": "v7"},
        }
        assert custom["child_queue"] == "default"

    def test_interface_is_declared(self):
        interface = _serialize(pipeline).interface
        assert [entry.key for entry in interface.inputs.variables] == ["x", "y"]
        assert [entry.key for entry in interface.outputs.variables] == ["o0"]

    def test_child_queue_is_omitted_when_unset(self):
        @flyte.sandbox.orchestrator
        def bare(x: int) -> int:
            return add(x, 1)

        assert "child_queue" not in MessageToDict(_serialize(bare).custom)
        assert bare.queue is None

    def test_serializing_before_tasks_are_resolved_is_an_error(self):
        @flyte.sandbox.orchestrator
        def unresolved(x: int) -> int:
            return add(x, 1)

        with pytest.raises(flyte.errors.RuntimeSystemError, match="add"):
            translate_task_to_wire(unresolved, SerializationContext(version="v1"))


class TestDecorator:
    def test_names_and_queue(self):
        assert pipeline.name == f"{__name__}.pipeline"
        assert pipeline.queue == "sandbox"
        assert not pipeline.runs_in_container()
        assert local_double.runs_in_container()

        @flyte.sandbox.orchestrator(name="custom.name")
        def named(x: int) -> int:
            return add(x, 1)

        assert named.name == "custom.name"

    def test_local_tasks_are_rejected(self):
        with pytest.raises(flyte.errors.RuntimeUserError, match=r"local_double is defined locally"):

            @flyte.sandbox.orchestrator
            def calls_local(x: int) -> int:
                return local_double(x)

    def test_version_follows_the_source(self):
        @flyte.sandbox.orchestrator
        def first(x: int) -> int:
            return add(x, 1)

        @flyte.sandbox.orchestrator
        def second(x: int) -> int:
            return add(x, 2)

        assert first.source_version == first.source_version
        assert first.source_version != second.source_version
