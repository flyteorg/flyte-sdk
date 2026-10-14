"""Caller-declared artifacts: `flyte.artifacts.produces`.

A caller names which outputs of the task it calls are artifacts. The declarations travel on the
child's Inputs.context under a reserved key, the child's output conversion turns them into
ProducedArtifact entries, and nothing is passed on to the actions the child spawns.
"""

from __future__ import annotations

import asyncio
from datetime import date

import pytest
from flyteidl2.task import task_definition_pb2

import flyte
import flyte.artifacts as artifacts
import flyte.report
from flyte._context import internal_ctx
from flyte._internal.runtime.convert import (
    PRODUCED_ARTIFACTS_CONTEXT_KEY,
    Inputs,
    convert_from_native_to_inputs,
    convert_from_native_to_outputs,
    decode_declared_artifacts,
)
from flyte.artifacts import Metadata
from flyte.errors import RuntimeUserError
from flyte.io import File
from flyte.models import ActionID, RawDataPath, TaskContext
from flyte.remote._task import TaskDetails

env = flyte.TaskEnvironment(name="produced-artifact-declarations-test")


@env.task
async def plain(x: int) -> File:
    return File(path=f"s3://bucket/{x}.pt")


@env.task
async def two_outputs() -> tuple[File, File]:
    return File(path="s3://bucket/a.pt"), File(path="s3://bucket/b.pt")


@env.task(produces_artifacts=True)
async def self_publishing() -> File:
    return File(path="s3://bucket/own.pt")


def _task_context(custom_context=None) -> TaskContext:
    return TaskContext(
        action=ActionID(name="a0", run_name="r1"),
        version="v1",
        raw_data_path=RawDataPath(path="s3://bucket/raw"),
        output_path="s3://bucket/out",
        run_base_dir="s3://bucket/base",
        report=flyte.report.Report(name="a0"),
        custom_context=dict(custom_context or {}),
    )


async def _child_inputs(**declared: Metadata) -> Inputs:
    """The Inputs a parent task would send to `plain(x=1)` inside `produces(**declared)`."""
    ctx = internal_ctx()
    with ctx.replace_task_context(_task_context({"team": "ml"})):
        with artifacts.produces(**declared):
            return await convert_from_native_to_inputs(plain.native_interface, x=1)


class TestDeclaringOutputs:
    @pytest.mark.asyncio
    async def test_declarations_ride_on_the_child_inputs(self):
        inputs = await _child_inputs(o0=Metadata(name="events", partitions={"date": date(2026, 9, 1), "region": "us"}))
        declared = inputs.declared_artifacts
        assert set(declared) == {"o0"}
        assert declared["o0"].name == "events"
        assert declared["o0"].partitions.value["region"].static_value == "us"
        assert declared["o0"].time_partition.key == "date"

    @pytest.mark.asyncio
    async def test_reserved_key_is_hidden_from_the_child_and_not_propagated(self):
        inputs = await _child_inputs(o0=Metadata(name="events"))
        # The child's user-facing context keeps ordinary keys and drops the reserved one ...
        assert inputs.context == {"team": "ml"}
        # ... so the actions the child spawns (built from that context) never see the declaration.
        ctx = internal_ctx()
        with ctx.replace_task_context(_task_context(inputs.context)):
            grandchild = await convert_from_native_to_inputs(plain.native_interface, x=2)
        assert grandchild.declared_artifacts == {}
        assert grandchild.context == {"team": "ml"}

    @pytest.mark.asyncio
    async def test_get_custom_context_hides_the_reserved_key(self):
        ctx = internal_ctx()
        with ctx.replace_task_context(_task_context({"team": "ml"})):
            with artifacts.produces(o0=Metadata(name="events")):
                assert flyte.get_custom_context() == {"team": "ml"}

    @pytest.mark.asyncio
    async def test_concurrent_calls_keep_their_own_declarations(self):
        async def one(name: str) -> str:
            await asyncio.sleep(0)
            inputs = await _child_inputs(o0=Metadata(name=name))
            await asyncio.sleep(0)
            return inputs.declared_artifacts["o0"].name

        assert await asyncio.gather(one("a"), one("b"), one("c")) == ["a", "b", "c"]

    def test_outside_a_task_it_does_nothing(self):
        with artifacts.produces(o0=Metadata(name="events")):
            assert flyte.get_custom_context() == {}

    def test_a_bad_declaration_fails_in_the_caller(self):
        with pytest.raises(Exception):
            with artifacts.produces(o0=Metadata(name="events", parents=("",))):
                pass


class TestProducingDeclaredOutputs:
    @pytest.mark.asyncio
    async def test_an_unwrapped_declared_output_is_produced(self):
        declared = (await _child_inputs(o0=Metadata(name="events", partitions={"region": "us"}))).declared_artifacts
        outputs = await convert_from_native_to_outputs(
            File(path="s3://bucket/1.pt"), plain.native_interface, "plain", declared=declared
        )
        (pa,) = outputs.proto_outputs.produced_artifacts
        assert pa.output == "o0"
        assert pa.name == "events"
        assert pa.partitions.value["region"].static_value == "us"
        assert pa.type.HasField("blob"), "the task fills in the literal type"

    @pytest.mark.asyncio
    async def test_only_declared_slots_are_produced(self):
        ctx = internal_ctx()
        with ctx.replace_task_context(_task_context()):
            with artifacts.produces(o1=Metadata(name="b")):
                inputs = await convert_from_native_to_inputs(two_outputs.native_interface)
        outputs = await convert_from_native_to_outputs(
            (File(path="s3://bucket/a.pt"), File(path="s3://bucket/b.pt")),
            two_outputs.native_interface,
            "two_outputs",
            declared=inputs.declared_artifacts,
        )
        assert [(pa.output, pa.name) for pa in outputs.proto_outputs.produced_artifacts] == [("o1", "b")]

    @pytest.mark.asyncio
    async def test_no_declaration_changes_nothing(self):
        outputs = await convert_from_native_to_outputs(File(path="s3://bucket/1.pt"), plain.native_interface, "plain")
        assert list(outputs.proto_outputs.produced_artifacts) == []

    @pytest.mark.asyncio
    async def test_caller_wins_identity_and_task_fills_the_rest(self):
        declared = (await _child_inputs(o0=Metadata(name="events", attrs={"flyte.io/factory": "f"}))).declared_artifacts
        own = artifacts.new(
            File(path="s3://bucket/own.pt"), Metadata(name="own", description="weights", attrs={"k": "v"})
        )
        outputs = await convert_from_native_to_outputs(
            own, self_publishing.native_interface, "self_publishing", declared=declared
        )
        (pa,) = outputs.proto_outputs.produced_artifacts
        assert pa.name == "events"
        assert pa.info.description == "weights"
        assert dict(pa.info.user_metadata) == {"k": "v", "flyte.io/factory": "f"}

    @pytest.mark.asyncio
    async def test_a_declaration_for_a_missing_output_is_an_error(self):
        declared = (await _child_inputs(o3=Metadata(name="nope"))).declared_artifacts
        with pytest.raises(RuntimeUserError, match="o3"):
            await convert_from_native_to_outputs(
                File(path="s3://bucket/1.pt"), plain.native_interface, "plain", declared=declared
            )


class TestRemoteOverride:
    def test_produces_artifacts_sets_the_template_flag(self):
        pb2 = task_definition_pb2.TaskDetails()
        pb2.spec.task_template.id.name = "t"
        details = TaskDetails(pb2)
        assert details.override(produces_artifacts=True).pb2.spec.task_template.metadata.produces_artifacts
        assert not details.override(produces_artifacts=False).pb2.spec.task_template.metadata.produces_artifacts
        assert not details.override().pb2.spec.task_template.metadata.produces_artifacts


def test_reserved_key_value_is_the_encoded_declarations():
    async def run():
        inputs = await _child_inputs(o0=Metadata(name="events"))
        raw = next(kv.value for kv in inputs.proto_inputs.context if kv.key == PRODUCED_ARTIFACTS_CONTEXT_KEY)
        return decode_declared_artifacts(raw)["o0"].name

    assert asyncio.run(run()) == "events"
