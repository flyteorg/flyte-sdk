"""Runtime lineage: a produced artifact records the artifact versions its action consumed as parents.

Artifact-bound inputs carry `Literal.artifact_id`, stamped by the artifact service. The task pod already holds
those literals, so collecting them costs nothing; they are appended to each produced artifact's parents after the
declared ones, de-duplicated and capped, unless the output's metadata sets `inherit_parents=False`.
"""

from __future__ import annotations

import pytest
from flyteidl2.core import artifact_id_pb2, literals_pb2
from flyteidl2.task import common_pb2

import flyte
import flyte.artifacts as artifacts
import flyte.report
from flyte._context import internal_ctx
from flyte._internal.runtime.convert import (
    Inputs,
    consumed_artifacts,
    convert_from_native_to_outputs,
)
from flyte.artifacts import Metadata
from flyte.artifacts._metadata import MAX_PARENTS
from flyte.io import File
from flyte.models import ActionID, RawDataPath, TaskContext

env = flyte.TaskEnvironment(name="inherited-parents-test")


@env.task
async def produce() -> File:
    return File(path="s3://bucket/out.pt")


@env.task
async def produce_two() -> tuple[File, File]:
    return File(path="s3://bucket/a.pt"), File(path="s3://bucket/b.pt")


def _vid(name: str, version: str, project: str = "p", domain: str = "d") -> artifact_id_pb2.ArtifactVersionId:
    return artifact_id_pb2.ArtifactVersionId(
        key=artifact_id_pb2.ArtifactKey(org="o", project=project, domain=domain, name=name), version=version
    )


def _artifact_literal(name: str, version: str) -> literals_pb2.Literal:
    lit = literals_pb2.Literal(scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(integer=1)))
    lit.artifact_id.CopyFrom(_vid(name, version))
    return lit


def _plain_literal() -> literals_pb2.Literal:
    return literals_pb2.Literal(scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(integer=7)))


def _inputs(**literals: literals_pb2.Literal) -> Inputs:
    return Inputs(
        proto_inputs=common_pb2.Inputs(literals=[common_pb2.NamedLiteral(name=k, value=v) for k, v in literals.items()])
    )


async def _convert(out, interface, consumed, *, scoped: bool = True):
    """Serialize `out` as the action o/p/d/r1/a0 would (or outside any task when not scoped)."""
    if not scoped:
        return await convert_from_native_to_outputs(out, interface, "t", consumed=consumed)
    tctx = TaskContext(
        action=ActionID(name="a0", run_name="r1", org="o", project="p", domain="d"),
        version="v1",
        raw_data_path=RawDataPath(path="s3://bucket/raw"),
        output_path="s3://bucket/out",
        run_base_dir="s3://bucket/base",
        report=flyte.report.Report(name="a0"),
    )
    with internal_ctx().replace_task_context(tctx):
        return await convert_from_native_to_outputs(out, interface, "t", consumed=consumed)


def _names(parents) -> list[str]:
    return [f"{p.key.name}@{p.version}" if p.key.name else f"@{p.version}" for p in parents]


class TestConsumedArtifacts:
    def test_artifact_bound_inputs_in_order(self):
        consumed = consumed_artifacts(
            _inputs(a=_artifact_literal("raw", "v1"), x=_plain_literal(), b=_artifact_literal("features", "v3"))
        )
        assert _names(consumed) == ["raw@v1", "features@v3"]

    def test_list_and_dict_elements(self):
        coll = literals_pb2.Literal(
            collection=literals_pb2.LiteralCollection(
                literals=[_artifact_literal("shard", "1"), _plain_literal(), _artifact_literal("shard", "2")]
            )
        )
        mp = literals_pb2.Literal(
            map=literals_pb2.LiteralMap(literals={"z": _artifact_literal("m", "b"), "a": _artifact_literal("m", "a")})
        )
        assert _names(consumed_artifacts(_inputs(xs=coll, m=mp))) == ["shard@1", "shard@2", "m@a", "m@b"]

    def test_duplicates_once(self):
        consumed = consumed_artifacts(_inputs(a=_artifact_literal("raw", "v1"), b=_artifact_literal("raw", "v1")))
        assert _names(consumed) == ["raw@v1"]

    def test_plain_inputs_and_no_inputs(self):
        assert consumed_artifacts(_inputs(x=_plain_literal())) == []
        assert consumed_artifacts(None) == []
        assert consumed_artifacts(Inputs.empty()) == []

    def test_an_artifact_literal_is_not_walked_into(self):
        # A collection bound as one artifact is one parent, not one per element.
        coll = literals_pb2.Literal(
            collection=literals_pb2.LiteralCollection(literals=[_artifact_literal("inner", "1")])
        )
        coll.artifact_id.CopyFrom(_vid("outer", "9"))
        assert _names(consumed_artifacts(_inputs(xs=coll))) == ["outer@9"]


class TestInheritedParents:
    @pytest.mark.asyncio
    async def test_produced_artifact_inherits_consumed_versions(self):
        out = artifacts.new(File(path="s3://bucket/model.pt"), Metadata(name="model"))
        consumed = [_vid("raw", "v1"), _vid("features", "v3")]
        outputs = await _convert(out, produce.native_interface, consumed)
        (pa,) = outputs.proto_outputs.produced_artifacts
        assert _names(pa.parent_artifacts) == ["raw@v1", "features@v3"]

    @pytest.mark.asyncio
    async def test_declared_parents_first_then_inherited_deduped(self):
        md = Metadata(name="model", parents=("v0", _vid("raw", "v1")))
        out = artifacts.new(File(path="s3://bucket/model.pt"), md)
        consumed = [_vid("raw", "v1"), _vid("features", "v3")]
        outputs = await _convert(out, produce.native_interface, consumed)
        (pa,) = outputs.proto_outputs.produced_artifacts
        assert _names(pa.parent_artifacts) == ["@v0", "raw@v1", "features@v3"]

    @pytest.mark.asyncio
    async def test_cap_keeps_declared_parents(self):
        declared = tuple(f"v{i}" for i in range(MAX_PARENTS - 2))
        out = artifacts.new(File(path="s3://bucket/model.pt"), Metadata(name="model", parents=declared))
        consumed = [_vid("in", str(i)) for i in range(10)]
        outputs = await _convert(out, produce.native_interface, consumed)
        (pa,) = outputs.proto_outputs.produced_artifacts
        assert len(pa.parent_artifacts) == MAX_PARENTS
        assert _names(pa.parent_artifacts)[: len(declared)] == [f"@{v}" for v in declared]
        assert _names(pa.parent_artifacts)[len(declared) :] == ["in@0", "in@1"]

    @pytest.mark.asyncio
    async def test_the_version_being_published_is_not_its_own_parent(self):
        out = artifacts.new(File(path="s3://bucket/model.pt"), Metadata(name="model", version="v5"))
        consumed = [_vid("model", "v5"), _vid("model", "v4")]
        outputs = await _convert(out, produce.native_interface, consumed)
        (pa,) = outputs.proto_outputs.produced_artifacts
        assert _names(pa.parent_artifacts) == ["model@v4"]

    @pytest.mark.asyncio
    async def test_opt_out_per_output(self):
        a = artifacts.new(File(path="s3://bucket/a.pt"), Metadata(name="a"))
        b = artifacts.new(File(path="s3://bucket/b.pt"), Metadata(name="b", inherit_parents=False))
        outputs = await _convert((a, b), produce_two.native_interface, [_vid("raw", "v1")])
        by_name = {pa.name: pa for pa in outputs.proto_outputs.produced_artifacts}
        assert _names(by_name["a"].parent_artifacts) == ["raw@v1"]
        assert list(by_name["b"].parent_artifacts) == []

    @pytest.mark.asyncio
    async def test_outputs_that_are_not_artifacts_are_untouched(self):
        outputs = await _convert(File(path="s3://bucket/x.pt"), produce.native_interface, [_vid("raw", "v1")])
        assert list(outputs.proto_outputs.produced_artifacts) == []

    @pytest.mark.asyncio
    async def test_only_the_actions_own_scope_is_inherited(self):
        # The artifact service rejects a parent in another org/project/domain, which would drop the artifact.
        out = artifacts.new(File(path="s3://bucket/model.pt"), Metadata(name="model"))
        consumed = [_vid("raw", "v1", project="other"), _vid("raw", "v2", domain="prod"), _vid("features", "v3")]
        outputs = await _convert(out, produce.native_interface, consumed)
        (pa,) = outputs.proto_outputs.produced_artifacts
        assert _names(pa.parent_artifacts) == ["features@v3"]
        # Written name-scoped: the service fills the child's org/project/domain.
        assert (pa.parent_artifacts[0].key.org, pa.parent_artifacts[0].key.project) == ("", "")

    @pytest.mark.asyncio
    async def test_nothing_is_inherited_without_a_known_scope(self):
        out = artifacts.new(File(path="s3://bucket/model.pt"), Metadata(name="model"))
        outputs = await _convert(out, produce.native_interface, [_vid("raw", "v1")], scoped=False)
        (pa,) = outputs.proto_outputs.produced_artifacts
        assert list(pa.parent_artifacts) == []

    @pytest.mark.asyncio
    async def test_dedupe_matches_the_service(self):
        # A bare declared version means "same name": consuming model@v0 must not add it again.
        out = artifacts.new(File(path="s3://bucket/model.pt"), Metadata(name="model", parents=("v0",)))
        outputs = await _convert(out, produce.native_interface, [_vid("model", "v0"), _vid("model", "v1")])
        (pa,) = outputs.proto_outputs.produced_artifacts
        assert _names(pa.parent_artifacts) == ["@v0", "model@v1"]

    @pytest.mark.asyncio
    async def test_nothing_consumed_changes_nothing(self):
        out = artifacts.new(File(path="s3://bucket/model.pt"), Metadata(name="model", parents=("v0",)))
        for consumed in (None, []):
            outputs = await _convert(out, produce.native_interface, consumed)
            (pa,) = outputs.proto_outputs.produced_artifacts
            assert _names(pa.parent_artifacts) == ["@v0"]
