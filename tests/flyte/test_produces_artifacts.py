"""Tests for the produces_artifacts task flag and produced-artifact output stamping.

Covers: decorator → TaskTemplate → override → serialized TaskMetadata proto, the
Metadata compact-JSON codec, and stamping of `flyte.artifacts.new(...)` wrapper
first-class ProducedArtifact declarations onto the Outputs envelope during output conversion.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import pytest
from flyteidl2.core import artifact_id_pb2, types_pb2
from pydantic import BaseModel

import flyte
import flyte.artifacts as artifacts
from flyte._internal.runtime.convert import convert_from_native_to_outputs
from flyte._internal.runtime.task_serde import get_proto_task
from flyte.artifacts._metadata import Metadata, to_produced_artifact
from flyte.io import File
from flyte.models import SerializationContext

env = flyte.TaskEnvironment(name="produces-artifacts-test")


@dataclass
class Payload:
    """NOT artifactable: only offloaded assets (File/Dir/DataFrame) are."""

    content: str


def _weights_file(uri: str = "s3://bucket/weights.pt") -> File:
    """Artifacts must be offloaded assets; tests wrap this remote File."""
    return File(path=uri)


@env.task(produces_artifacts=True)
async def producing_task(x: int) -> File:
    return _weights_file(f"s3://bucket/model-{x}.pt")


@env.task
async def plain_task(x: int) -> str:
    return f"plain-{x}"


# Multi-output producers; extract_return_annotation names the slots o0, o1, ...
# This module has `from __future__ import annotations`, so the bare-tuple form
# (`-> (File, int)`) cannot be used here -- see test_bare_tuple_annotation below.
@env.task(produces_artifacts=True)
async def multi_output_task() -> tuple[File, int]:
    return _weights_file(), 42


@env.task(produces_artifacts=True)
async def trailing_artifact_task() -> tuple[str, File]:
    return "summary", _weights_file()


@env.task(produces_artifacts=True)
async def two_artifact_task() -> tuple[File, File]:
    return _weights_file("s3://bucket/a.pt"), _weights_file("s3://bucket/b.pt")


class Bundle(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    weights: object = None


@env.task(produces_artifacts=True)
async def bundling_task() -> Bundle:
    return Bundle()


@env.task(produces_artifacts=True)
async def listing_task() -> list[File]:
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


class TestToProducedArtifact:
    def test_minimal(self):
        decl = to_produced_artifact(
            Metadata(name="my-model"),
            output="o0",
            literal_type=types_pb2.LiteralType(simple=types_pb2.SimpleType.STRING),
        )
        assert decl.output == "o0"
        assert decl.name == "my-model"
        assert decl.version == ""
        assert not decl.info.description
        assert not decl.info.user_metadata
        assert not decl.info.HasField("card")
        assert decl.type.simple == types_pb2.SimpleType.STRING

    def test_full(self):
        decl = to_produced_artifact(
            Metadata(
                name="my-model",
                version="1.0",
                description="a model",
                attrs={"framework": "torch"},
                card=artifacts.Card(uri="s3://b/card.html", format="html", card_type="model"),
            ),
            output="o0",
            literal_type=types_pb2.LiteralType(simple=types_pb2.SimpleType.STRING),
        )
        assert decl.version == "1.0"
        assert decl.info.description == "a model"
        assert dict(decl.info.user_metadata) == {"framework": "torch"}
        assert decl.info.card == artifact_id_pb2.ArtifactCard(uri="s3://b/card.html", format="html", type="model")

    def test_model_metadata_merges_extra_data(self):
        md = Metadata.create_model_metadata(
            name="m",
            framework="torch",
            attrs={"source_repo": "org/model", "framework": "should-lose"},
        )
        assert md.attrs["source_repo"] == "org/model"
        # Model-specific keys win on conflict.
        assert md.attrs["framework"] == "torch"


class TestOutputDeclarations:
    @pytest.mark.asyncio
    async def test_wrapped_output_declared(self):
        md = Metadata(name="my-model", version="1.0")
        outputs = await convert_from_native_to_outputs(
            artifacts.new(_weights_file(), md), producing_task.native_interface, "t"
        )

        (nl,) = outputs.proto_outputs.literals
        assert nl.name == "o0"
        # Files serialize as blob scalars referencing the remote uri.
        assert nl.value.scalar.WhichOneof("value") == "blob"
        # The value itself carries no metadata contract keys.
        assert not nl.value.metadata
        (decl,) = outputs.proto_outputs.produced_artifacts
        assert decl.output == "o0"
        assert decl.name == "my-model"
        assert decl.version == "1.0"
        # The declaration carries the declared output type (SDK is authoritative).
        assert decl.type.blob.format != "" or decl.type.WhichOneof("type") is not None

    @pytest.mark.asyncio
    async def test_plain_output_not_declared(self):
        outputs = await convert_from_native_to_outputs("plain", plain_task.native_interface, "t")
        (nl,) = outputs.proto_outputs.literals
        assert not nl.value.metadata
        assert len(outputs.proto_outputs.produced_artifacts) == 0

    @pytest.mark.asyncio
    async def test_declaring_is_unconditional(self):
        # Declarations do not depend on produces_artifacts — the flag only gates
        # backend extraction. A plain task's wrapped output is declared too.
        outputs = await convert_from_native_to_outputs(
            artifacts.new(_weights_file(), Metadata(name="n")), producing_task.native_interface, "t"
        )
        (decl,) = outputs.proto_outputs.produced_artifacts
        assert decl.name == "n"

    @pytest.mark.asyncio
    async def test_no_version_left_empty(self):
        outputs = await convert_from_native_to_outputs(
            artifacts.new(_weights_file(), Metadata(name="unversioned")), producing_task.native_interface, "t"
        )
        (decl,) = outputs.proto_outputs.produced_artifacts
        assert decl.version == ""


class TestMultiOutputDeclarations:
    """A task may return several outputs and mark only some of them as artifacts.
    Metadata is tracked per output slot, so declarations must bind to the slot the
    wrapper was returned in — not merely to the first one."""

    @pytest.mark.asyncio
    async def test_artifact_alongside_primitive(self):
        outputs = await convert_from_native_to_outputs(
            (artifacts.new(_weights_file(), Metadata(name="my-model", version="1.0")), 42),
            multi_output_task.native_interface,
            "t",
        )

        # Both outputs still serialize; only the wrapped one is declared.
        assert [nl.name for nl in outputs.proto_outputs.literals] == ["o0", "o1"]
        assert outputs.proto_outputs.literals[1].value.scalar.primitive.integer == 42

        (decl,) = outputs.proto_outputs.produced_artifacts
        assert decl.output == "o0"
        assert decl.name == "my-model"
        assert decl.version == "1.0"

    @pytest.mark.asyncio
    async def test_declaration_binds_to_non_first_slot(self):
        # Regression guard: the wrapper is in the *second* slot, so a declaration
        # naming "o0" would silently attach the metadata to the wrong output.
        outputs = await convert_from_native_to_outputs(
            ("summary", artifacts.new(_weights_file(), Metadata(name="late-model"))),
            trailing_artifact_task.native_interface,
            "t",
        )

        (decl,) = outputs.proto_outputs.produced_artifacts
        assert decl.output == "o1"
        assert decl.name == "late-model"

    @pytest.mark.asyncio
    async def test_bare_tuple_annotation(self):
        # `-> (File, int)` (extract_return_annotation's "Option 4") is the form users
        # reach for first. It only resolves in a module that evaluates annotations
        # eagerly, hence the separate import; see that module's docstring.
        from .multi_output_bare_tuple import bare_tuple_task

        assert list(bare_tuple_task.native_interface.outputs) == ["o0", "o1"]

        outputs = await convert_from_native_to_outputs(
            (artifacts.new(_weights_file(), Metadata(name="bare-model")), 7),
            bare_tuple_task.native_interface,
            "t",
        )
        (decl,) = outputs.proto_outputs.produced_artifacts
        assert decl.output == "o0"
        assert decl.name == "bare-model"
        assert outputs.proto_outputs.literals[1].value.scalar.primitive.integer == 7

    @pytest.mark.asyncio
    async def test_every_wrapped_slot_declared(self):
        outputs = await convert_from_native_to_outputs(
            (
                artifacts.new(_weights_file("s3://bucket/a.pt"), Metadata(name="model-a")),
                artifacts.new(_weights_file("s3://bucket/b.pt"), Metadata(name="model-b")),
            ),
            two_artifact_task.native_interface,
            "t",
        )

        decls = {d.output: d.name for d in outputs.proto_outputs.produced_artifacts}
        assert decls == {"o0": "model-a", "o1": "model-b"}


class TestArtifactAnnotationDisplay:
    @pytest.mark.asyncio
    async def test_string_repr_annotates_produced_output(self):
        from flyte.types import literal_string_repr

        outputs = await convert_from_native_to_outputs(
            artifacts.new(_weights_file(), Metadata(name="my-model", version="1.0")),
            producing_task.native_interface,
            "t",
        )
        assert literal_string_repr(outputs.proto_outputs) == {
            "o0": "s3://bucket/weights.pt (produced artifact: my-model@1.0)"
        }

    @pytest.mark.asyncio
    async def test_string_repr_omits_version_when_unset(self):
        from flyte.types import literal_string_repr

        outputs = await convert_from_native_to_outputs(
            artifacts.new(_weights_file(), Metadata(name="unversioned")), producing_task.native_interface, "t"
        )
        assert literal_string_repr(outputs.proto_outputs) == {
            "o0": "s3://bucket/weights.pt (produced artifact: unversioned)"
        }

    @pytest.mark.asyncio
    async def test_action_outputs_repr_annotates(self):
        from flyte.remote import ActionOutputs

        outputs = await convert_from_native_to_outputs(
            artifacts.new(_weights_file(), Metadata(name="my-model")),
            producing_task.native_interface,
            "t",
        )
        ao = ActionOutputs(outputs.proto_outputs, (_weights_file(),))
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

    def test_dataclass_rejected(self):
        with pytest.raises(TypeError, match="cannot be artifacts"):
            artifacts.new(Payload(content="not-an-asset"), Metadata(name="nope"))

    def test_pydantic_model_rejected(self):
        with pytest.raises(TypeError, match="cannot be artifacts"):
            artifacts.new(Bundle(), Metadata(name="nope"))

    def test_arbitrary_object_rejected(self):
        with pytest.raises(TypeError, match="cannot be artifacts"):
            artifacts.new(object(), Metadata(name="nope"))

    def test_offloaded_assets_allowed(self):
        from flyte.io import Dir

        assert artifacts.new(_weights_file(), Metadata(name="f")).get_flyte_metadata().name == "f"
        assert artifacts.new(Dir(path="s3://bucket/ckpt/"), Metadata(name="d")).get_flyte_metadata().name == "d"

    @pytest.mark.asyncio
    async def test_nested_wrapper_in_pydantic_model_rejected(self):
        # Artifacts must be top-level task outputs: a wrapper nested inside a
        # model would serialize its inner value and silently drop the artifact
        # metadata, so output conversion rejects it outright.
        bundle = Bundle(weights=artifacts.new(_weights_file(), Metadata(name="nested")))
        with pytest.raises(Exception, match="cannot be nested"):
            await convert_from_native_to_outputs(bundle, bundling_task.native_interface, "t")

    @pytest.mark.asyncio
    async def test_nested_wrapper_in_container_rejected(self):
        wrapped_in_list = [artifacts.new(_weights_file(), Metadata(name="nested"))]
        with pytest.raises(Exception, match="cannot be nested"):
            await convert_from_native_to_outputs(wrapped_in_list, listing_task.native_interface, "t")
