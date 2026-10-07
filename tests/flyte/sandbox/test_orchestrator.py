"""Tests for `flyte.sandbox.orchestrator`, whose source runs on a sandbox leaseworker."""

import asyncio
import base64
from types import SimpleNamespace

import pytest
from flyteidl2.core import tasks_pb2
from flyteidl2.task import task_definition_pb2
from google.protobuf.json_format import MessageToDict

import flyte
import flyte.errors
import flyte.sandbox
from flyte._internal.runtime.task_serde import translate_task_to_wire
from flyte.durable import sleep as durable_sleep
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


@flyte.sandbox.orchestrator(queue="gpu-pool")
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
        assert template.worker_kind == tasks_pb2.WorkerKind.WORKER_KIND_SANDBOX
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
        assert set(custom) == {"source", "input_names", "tasks", "timeout_ms", "max_stack_depth"}

    def test_template_carries_the_limits(self):
        @flyte.sandbox.orchestrator(timeout_ms=3_000, max_stack_depth=64)
        def limited(x: int) -> int:
            return add(x, 1)

        custom = MessageToDict(_serialize(limited).custom)
        assert custom["timeout_ms"] == 3000
        assert custom["max_stack_depth"] == 64

    def test_interface_is_declared(self):
        interface = _serialize(pipeline).interface
        assert [entry.key for entry in interface.inputs.variables] == ["x", "y"]
        assert [entry.key for entry in interface.outputs.variables] == ["o0"]

    def test_queue_is_optional(self):
        @flyte.sandbox.orchestrator
        def bare(x: int) -> int:
            return add(x, 1)

        _serialize(bare)
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
        assert pipeline.queue == "gpu-pool"
        assert not pipeline.runs_in_container
        assert local_double.runs_in_container
        assert pipeline.worker_kind == "sandbox"
        assert local_double.worker_kind == ""

        @flyte.sandbox.orchestrator(name="custom.name")
        def named(x: int) -> int:
            return add(x, 1)

        assert named.name == "custom.name"

    def test_version_follows_the_source(self):
        @flyte.sandbox.orchestrator
        def first(x: int) -> int:
            return add(x, 1)

        @flyte.sandbox.orchestrator
        def second(x: int) -> int:
            return add(x, 2)

        assert first.source_version == first.source_version
        assert first.source_version != second.source_version


class TestLocalTasks:
    """Local tasks have no deployed spec, so theirs travels in the template."""

    @staticmethod
    def _serialize(task: OrchestratorTaskTemplate):
        async def serialize_local(local_task):
            spec = task_definition_pb2.TaskSpec()
            spec.task_template.id.project = "proj"
            spec.task_template.id.domain = "dev"
            spec.task_template.id.name = local_task.name
            spec.task_template.id.version = "bundle-v3"
            spec.task_template.type = "python"
            return spec

        asyncio.run(task.resolve_tasks(serialize_local))
        template = translate_task_to_wire(task, SerializationContext(version="v1")).task_template
        return MessageToDict(template.custom)

    def test_local_task_spec_travels_in_the_template(self):
        @flyte.sandbox.orchestrator
        def calls_local(x: int) -> int:
            return local_double(x)

        entry = self._serialize(calls_local)["tasks"]["local_double"]
        assert {k: entry[k] for k in ("project", "domain", "name", "version")} == {
            "project": "proj",
            "domain": "dev",
            "name": "local-env.local_double",
            "version": "bundle-v3",
        }
        spec = task_definition_pb2.TaskSpec()
        spec.ParseFromString(base64.b64decode(entry["spec"]))
        assert spec.task_template.id.name == "local-env.local_double"
        assert spec.task_template.type == "python"

    def test_remote_and_local_tasks_can_be_mixed(self):
        @flyte.sandbox.orchestrator
        def mixed(x: int) -> int:
            return add(local_double(x), 1)

        tasks = self._serialize(mixed)["tasks"]
        assert "spec" in tasks["local_double"]
        # A remote task is fetched by the worker, so only its id is shipped.
        assert tasks["add"] == {"project": "proj", "domain": "dev", "name": "math.add", "version": "v7"}

    def test_local_task_cannot_be_resolved_without_a_serializer(self):
        @flyte.sandbox.orchestrator
        def calls_local(x: int) -> int:
            return local_double(x)

        with pytest.raises(flyte.errors.RuntimeSystemError, match="local_double"):
            asyncio.run(calls_local.resolve_tasks())

    def test_version_follows_the_tasks_it_calls(self):
        @flyte.sandbox.orchestrator
        def calls_local(x: int) -> int:
            return local_double(x)

        before = calls_local.source_version
        self._serialize(calls_local)
        assert calls_local.source_version != before


class TestUnsupportedRefs:
    def test_durable_operations_are_rejected(self):
        with pytest.raises(flyte.errors.RuntimeUserError, match=r"durable_sleep is not"):

            @flyte.sandbox.orchestrator
            def calls_durable(x: int) -> int:
                durable_sleep(1)
                return add(x, 1)
