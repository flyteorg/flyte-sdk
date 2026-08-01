"""Tests for the produces_artifacts task flag and produced-artifact output stamping.

Covers: decorator → TaskTemplate → override → serialized TaskMetadata proto, the
Metadata compact-JSON codec, and stamping of `flyte.artifacts.new(...)` wrapper
metadata onto output literals (under ARTIFACT_PRODUCED_KEY) during output conversion.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

import flyte
import flyte.artifacts as artifacts
from flyte._constants import ARTIFACT_PRODUCED_KEY
from flyte._internal.runtime.convert import convert_from_native_to_outputs
from flyte._internal.runtime.task_serde import get_proto_task
from flyte.artifacts._metadata import Metadata, from_compact_json, to_compact_json
from flyte.models import SerializationContext

env = flyte.TaskEnvironment(name="produces-artifacts-test")


@dataclass
class Payload:
    """Artifacts must be non-primitive; tests wrap this payload."""

    content: str


@env.task(produces_artifacts=True)
async def producing_task(x: int) -> Payload:
    return Payload(content=f"model-{x}")


@env.task
async def plain_task(x: int) -> str:
    return f"plain-{x}"


class Bundle(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    weights: object = None


@env.task(produces_artifacts=True)
async def bundling_task() -> Bundle:
    return Bundle()


@env.task(produces_artifacts=True)
async def listing_task() -> list[Payload]:
    return []


def _serialization_context() -> SerializationContext:
    return SerializationContext(
        project="test-project",
        domain="test-domain",
        version="test-version",
        org="test-org",
        input_path="/tmp/inputs",
        output_path="/tmp/outputs",
        image_cache=None,
        code_bundle=None,
        root_dir=pathlib.Path.cwd(),
    )


class TestFlagPlumbing:
    def test_decorator_sets_flag(self):
        assert producing_task.produces_artifacts is True
        assert plain_task.produces_artifacts is False

    def test_override_toggles_flag(self):
        assert plain_task.override(produces_artifacts=True).produces_artifacts is True
        assert producing_task.override(produces_artifacts=False).produces_artifacts is False
        # Unspecified override preserves the original value.
        assert producing_task.override(retries=1).produces_artifacts is True

    def test_serde_carries_flag(self):
        proto = get_proto_task(producing_task, _serialization_context())
        assert proto.metadata.produces_artifacts is True

        proto = get_proto_task(plain_task, _serialization_context())
        assert proto.metadata.produces_artifacts is False


class TestMetadataCompactJson:
    def test_minimal_round_trip(self):
        md = Metadata(name="my-model")
        s = to_compact_json(md)
        # Pinned fixture: the Go-side reader (leaseworker/artifacts.go) parses this shape.
        assert s == '{"name":"my-model"}'
        assert from_compact_json(s) == md

    def test_full_round_trip_pinned_fixture(self):
        md = Metadata(
            name="my-model",
            version="1.0",
            description="a model",
            data={"framework": "torch"},
            card=artifacts.Card(uri="s3://b/card.html", format="html", card_type="model"),
        )
        s = to_compact_json(md)
        # Pinned fixture shared (byte-identical) with the Go test in leaseworker/artifacts_test.go.
        assert s == (
            '{"card":{"format":"html","type":"model","uri":"s3://b/card.html"},'
            '"data":{"framework":"torch"},"description":"a model","name":"my-model","version":"1.0"}'
        )
        assert from_compact_json(s) == md

    def test_deterministic(self):
        md = Metadata(name="n", data={"b": "2", "a": "1"})
        assert to_compact_json(md) == to_compact_json(Metadata(name="n", data={"a": "1", "b": "2"}))


class TestOutputStamping:
    @pytest.mark.asyncio
    async def test_wrapped_output_is_stamped(self):
        md = Metadata(name="my-model", version="1.0")
        outputs = await convert_from_native_to_outputs(
            artifacts.new(Payload(content="model-bytes"), md), producing_task.native_interface, "t"
        )

        (nl,) = outputs.proto_outputs.literals
        assert nl.name == "o0"
        # Non-primitive payloads serialize as binary (msgpack) scalars.
        assert nl.value.scalar.WhichOneof("value") == "binary"
        stamped = json.loads(nl.value.metadata[ARTIFACT_PRODUCED_KEY])
        assert stamped == {"name": "my-model", "version": "1.0"}
        assert from_compact_json(nl.value.metadata[ARTIFACT_PRODUCED_KEY]) == md

    @pytest.mark.asyncio
    async def test_plain_output_not_stamped(self):
        outputs = await convert_from_native_to_outputs("plain", plain_task.native_interface, "t")
        (nl,) = outputs.proto_outputs.literals
        assert ARTIFACT_PRODUCED_KEY not in nl.value.metadata

    @pytest.mark.asyncio
    async def test_stamping_is_unconditional(self):
        # Stamping does not depend on produces_artifacts — the flag only gates
        # backend extraction. A plain task's wrapped output is stamped too.
        outputs = await convert_from_native_to_outputs(
            artifacts.new(Payload(content="v"), Metadata(name="n")), producing_task.native_interface, "t"
        )
        (nl,) = outputs.proto_outputs.literals
        assert nl.value.metadata[ARTIFACT_PRODUCED_KEY] == '{"name":"n"}'

    @pytest.mark.asyncio
    async def test_no_version_omitted_from_json(self):
        outputs = await convert_from_native_to_outputs(
            artifacts.new(Payload(content="v"), Metadata(name="unversioned")), producing_task.native_interface, "t"
        )
        (nl,) = outputs.proto_outputs.literals
        assert "version" not in json.loads(nl.value.metadata[ARTIFACT_PRODUCED_KEY])


class TestArtifactAnnotationDisplay:
    @pytest.mark.asyncio
    async def test_string_repr_annotates_produced_output(self):
        from flyte.types import literal_string_repr

        outputs = await convert_from_native_to_outputs(
            artifacts.new(Payload(content="model-bytes"), Metadata(name="my-model", version="1.0")),
            producing_task.native_interface,
            "t",
        )
        assert literal_string_repr(outputs.proto_outputs) == {
            "o0": '{"content": "model-bytes"} (produced artifact: my-model@1.0)'
        }

    @pytest.mark.asyncio
    async def test_string_repr_omits_version_when_unset(self):
        from flyte.types import literal_string_repr

        outputs = await convert_from_native_to_outputs(
            artifacts.new(Payload(content="v"), Metadata(name="unversioned")), producing_task.native_interface, "t"
        )
        assert literal_string_repr(outputs.proto_outputs) == {"o0": '{"content": "v"} (produced artifact: unversioned)'}

    @pytest.mark.asyncio
    async def test_action_outputs_repr_annotates(self):
        from flyte.remote import ActionOutputs

        outputs = await convert_from_native_to_outputs(
            artifacts.new(Payload(content="model-bytes"), Metadata(name="my-model")),
            producing_task.native_interface,
            "t",
        )
        ao = ActionOutputs(outputs.proto_outputs, (Payload(content="model-bytes"),))
        assert "(produced artifact: my-model)" in repr(ao)

    @pytest.mark.asyncio
    async def test_action_outputs_repr_plain(self):
        from flyte.remote import ActionOutputs

        outputs = await convert_from_native_to_outputs("plain", plain_task.native_interface, "t")
        ao = ActionOutputs(outputs.proto_outputs, ("plain",))
        assert repr(ao) == 'ActionOutputs(o0="plain")'


class TestArtifactTypeRestrictions:
    @pytest.mark.parametrize("value", ["s", 1, 1.5, True, b"bytes", None])
    def test_primitives_rejected(self, value):
        with pytest.raises(TypeError, match="cannot be artifacts"):
            artifacts.new(value, Metadata(name="nope"))

    def test_structured_values_allowed(self):
        wrapped = artifacts.new(Payload(content="ok"), Metadata(name="ok"))
        assert wrapped.get_flyte_metadata().name == "ok"

    @pytest.mark.asyncio
    async def test_nested_wrapper_in_pydantic_model_rejected(self):
        # Artifacts must be top-level task outputs: a wrapper nested inside a
        # model would serialize its inner value and silently drop the artifact
        # metadata, so output conversion rejects it outright.
        bundle = Bundle(weights=artifacts.new(Payload(content="w"), Metadata(name="nested")))
        with pytest.raises(Exception, match="cannot be nested"):
            await convert_from_native_to_outputs(bundle, bundling_task.native_interface, "t")

    @pytest.mark.asyncio
    async def test_nested_wrapper_in_container_rejected(self):
        wrapped_in_list = [artifacts.new(Payload(content="w"), Metadata(name="nested"))]
        with pytest.raises(Exception, match="cannot be nested"):
            await convert_from_native_to_outputs(wrapped_in_list, listing_task.native_interface, "t")
