"""Tests for flyte.remote.Artifact create/get/listall against a fake ArtifactService."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.artifact import artifact_pb2, artifact_service_pb2
from flyteidl2.common import list_pb2

import flyte.artifacts as artifacts
from flyte.io import File
from flyte.remote._artifact import Artifact
from flyte.types import TypeEngine


def _cfg():
    return MagicMock(org="test-org", project="proj", domain="dev")


def _payload(uri: str = "s3://bucket/weights.pt") -> File:
    """Artifacts must be offloaded assets; tests publish this remote File."""
    return File(path=uri)


async def _stored_artifact(data: str, name: str = "my_artifact", version: str = "1.0") -> artifact_pb2.Artifact:
    lt = TypeEngine.to_literal_type(str)
    lit = await TypeEngine.to_literal(data, str, lt)
    return artifact_pb2.Artifact(
        artifact_id=artifact_pb2.ArtifactIdentifier(
            name=artifact_pb2.ArtifactName(org="test-org", project="proj", domain="dev", name=name),
            version=version,
        ),
        spec=artifact_pb2.ArtifactSpec(value=lit, type=lt),
    )


def _patched(client):
    return (
        patch("flyte.remote._artifact.ensure_client"),
        patch("flyte.remote._artifact.get_init_config", return_value=_cfg()),
        patch("flyte.remote._artifact.get_client", return_value=client),
    )


class TestCreate:
    @pytest.mark.asyncio
    async def test_create_plain_value(self):
        client = MagicMock()
        client.artifact_service.create_artifact = AsyncMock(
            return_value=artifact_service_pb2.CreateArtifactResponse(artifact=await _stored_artifact("hello"))
        )

        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            result = await Artifact.create.aio(_payload(), name="my_artifact", version="1.0", description="desc")

        req = client.artifact_service.create_artifact.await_args[0][0]
        assert req.artifact_id.name.org == "test-org"
        assert req.artifact_id.name.project == "proj"
        assert req.artifact_id.name.domain == "dev"
        assert req.artifact_id.name.name == "my_artifact"
        assert req.artifact_id.version == "1.0"
        assert req.spec.info.description == "desc"
        # The value round-trips through the type engine.
        assert (await TypeEngine.to_python_value(req.spec.value, File)).path == _payload().path
        assert result.name == "my_artifact"
        assert result.version == "1.0"

    @pytest.mark.asyncio
    async def test_create_from_wrapper_metadata(self):
        client = MagicMock()
        client.artifact_service.create_artifact = AsyncMock(
            return_value=artifact_service_pb2.CreateArtifactResponse(artifact=await _stored_artifact("v"))
        )
        md = artifacts.Metadata(
            name="wrapped",
            version="2.0",
            description="from metadata",
            attrs={"framework": "torch"},
            card=artifacts.Card(uri="s3://b/card.html", format="html", card_type="model"),
        )

        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            await Artifact.create.aio(artifacts.new(_payload(), md))

        req = client.artifact_service.create_artifact.await_args[0][0]
        assert req.artifact_id.name.name == "wrapped"
        assert req.artifact_id.version == "2.0"
        assert req.spec.info.description == "from metadata"
        assert dict(req.spec.info.user_metadata) == {"framework": "torch"}
        assert req.spec.info.card.uri == "s3://b/card.html"
        assert req.spec.info.card.type == "model"

    @pytest.mark.asyncio
    async def test_create_defaults_random_version(self):
        client = MagicMock()
        client.artifact_service.create_artifact = AsyncMock(
            return_value=artifact_service_pb2.CreateArtifactResponse(artifact=await _stored_artifact("v"))
        )

        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            await Artifact.create.aio(_payload(), name="unversioned")

        req = client.artifact_service.create_artifact.await_args[0][0]
        assert req.artifact_id.version  # non-empty random version

    @pytest.mark.asyncio
    async def test_create_primitive_value_rejected(self):
        client = MagicMock()
        p1, p2, p3 = _patched(client)
        with p1, p2, p3, pytest.raises(TypeError, match="cannot be artifacts"):
            await Artifact.create.aio("a plain string", name="nope")

    @pytest.mark.asyncio
    async def test_create_non_asset_object_rejected(self):
        from dataclasses import dataclass

        @dataclass
        class Model:
            content: str

        client = MagicMock()
        p1, p2, p3 = _patched(client)
        with p1, p2, p3, pytest.raises(TypeError, match="cannot be artifacts"):
            await Artifact.create.aio(Model(content="not-an-asset"), name="nope")

    @pytest.mark.asyncio
    async def test_create_without_name_raises(self):
        client = MagicMock()
        p1, p2, p3 = _patched(client)
        with p1, p2, p3, pytest.raises(ValueError, match="name is required"):
            await Artifact.create.aio(_payload())

    @pytest.mark.asyncio
    async def test_create_with_external_ref_source(self):
        client = MagicMock()
        client.artifact_service.create_artifact = AsyncMock(
            return_value=artifact_service_pb2.CreateArtifactResponse(artifact=await _stored_artifact("v"))
        )

        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            await Artifact.create.aio(_payload(), name="imported", external_ref="hf://meta-llama/Meta-Llama-3-8B")

        req = client.artifact_service.create_artifact.await_args[0][0]
        assert req.spec.source.WhichOneof("source") == "external_ref"
        assert req.spec.source.external_ref == "hf://meta-llama/Meta-Llama-3-8B"

    @pytest.mark.asyncio
    async def test_create_outside_task_has_no_source(self):
        client = MagicMock()
        client.artifact_service.create_artifact = AsyncMock(
            return_value=artifact_service_pb2.CreateArtifactResponse(artifact=await _stored_artifact("v"))
        )

        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            await Artifact.create.aio(_payload(), name="manual")

        req = client.artifact_service.create_artifact.await_args[0][0]
        assert req.spec.source.WhichOneof("source") is None

    @pytest.mark.asyncio
    async def test_create_in_task_stamps_task_action_source(self):
        from flyte._context import internal_ctx
        from flyte.models import ActionID

        client = MagicMock()
        client.artifact_service.create_artifact = AsyncMock(
            return_value=artifact_service_pb2.CreateArtifactResponse(artifact=await _stored_artifact("v"))
        )

        tctx = MagicMock()
        tctx.action = ActionID(name="a0", run_name="r1", project="proj", domain="dev", org="test-org")
        tctx.attempt_number = 3

        ctx = internal_ctx()
        p1, p2, p3 = _patched(client)
        with p1, p2, p3, ctx.replace_task_context(tctx):
            await Artifact.create.aio(_payload(), name="produced")

        req = client.artifact_service.create_artifact.await_args[0][0]
        source = req.spec.source
        assert source.WhichOneof("source") == "task_action"
        # Scope fields are left empty; the server inherits the artifact's scope.
        assert source.task_action.action.run.name == "r1"
        assert source.task_action.action.name == "a0"
        assert source.task_action.attempt == 3

    @pytest.mark.asyncio
    async def test_external_ref_wins_over_task_context(self):
        from flyte._context import internal_ctx
        from flyte.models import ActionID

        client = MagicMock()
        client.artifact_service.create_artifact = AsyncMock(
            return_value=artifact_service_pb2.CreateArtifactResponse(artifact=await _stored_artifact("v"))
        )

        tctx = MagicMock()
        tctx.action = ActionID(name="a0", run_name="r1")
        tctx.attempt_number = 0

        ctx = internal_ctx()
        p1, p2, p3 = _patched(client)
        with p1, p2, p3, ctx.replace_task_context(tctx):
            await Artifact.create.aio(_payload(), name="imported", external_ref="s3://elsewhere/model")

        req = client.artifact_service.create_artifact.await_args[0][0]
        assert req.spec.source.WhichOneof("source") == "external_ref"


class TestSourceDisplay:
    @pytest.mark.asyncio
    async def test_source_property_task_action(self):
        pb2 = await _stored_artifact("v")
        pb2.spec.source.task_action.action.run.name = "r1"
        pb2.spec.source.task_action.action.name = "a0"
        pb2.spec.source.task_action.attempt = 2
        assert Artifact(pb2=pb2).source == "run r1/a0 (attempt 2)"

    @pytest.mark.asyncio
    async def test_source_property_external_ref(self):
        pb2 = await _stored_artifact("v")
        pb2.spec.source.external_ref = "hf://org/model"
        assert Artifact(pb2=pb2).source == "hf://org/model"

    @pytest.mark.asyncio
    async def test_source_property_empty(self):
        pb2 = await _stored_artifact("v")
        assert Artifact(pb2=pb2).source == ""


class TestGet:
    @pytest.mark.asyncio
    async def test_get_latest_omits_version(self):
        client = MagicMock()
        client.artifact_service.get_artifact = AsyncMock(
            return_value=artifact_service_pb2.GetArtifactResponse(artifact=await _stored_artifact("hello"))
        )

        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            result = await Artifact.get.aio("my_artifact")

        req = client.artifact_service.get_artifact.await_args[0][0]
        assert req.name.name == "my_artifact"
        assert not req.HasField("version")
        assert await result.to_python() == "hello"

    @pytest.mark.asyncio
    async def test_get_pinned_version(self):
        client = MagicMock()
        client.artifact_service.get_artifact = AsyncMock(
            return_value=artifact_service_pb2.GetArtifactResponse(artifact=await _stored_artifact("hello"))
        )

        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            await Artifact.get.aio("my_artifact", version="1.0")

        req = client.artifact_service.get_artifact.await_args[0][0]
        assert req.version == "1.0"


class TestListall:
    @pytest.mark.asyncio
    async def test_paginates_until_token_empty(self):
        page1 = artifact_service_pb2.ListArtifactsResponse(
            artifacts=[await _stored_artifact("a", version="3"), await _stored_artifact("b", version="2")],
            token="2",
        )
        page2 = artifact_service_pb2.ListArtifactsResponse(
            artifacts=[await _stored_artifact("c", version="1")],
            token="",
        )
        client = MagicMock()
        client.artifact_service.list_artifacts = AsyncMock(side_effect=[page1, page2])

        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            results = [a async for a in Artifact.listall.aio(name="my_artifact")]

        assert [a.version for a in results] == ["3", "2", "1"]
        first_req = client.artifact_service.list_artifacts.await_args_list[0][0][0]
        assert first_req.name == "my_artifact"
        assert first_req.project_id.organization == "test-org"
        second_req = client.artifact_service.list_artifacts.await_args_list[1][0][0]
        assert second_req.request.token == "2"

    @pytest.mark.asyncio
    async def test_limit_stops_early(self):
        page = artifact_service_pb2.ListArtifactsResponse(
            artifacts=[await _stored_artifact("a"), await _stored_artifact("b")],
            token="2",
        )
        client = MagicMock()
        client.artifact_service.list_artifacts = AsyncMock(return_value=page)

        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            results = [a async for a in Artifact.listall.aio(limit=2)]

        assert len(results) == 2
        client.artifact_service.list_artifacts.assert_awaited_once()
        req = client.artifact_service.list_artifacts.await_args[0][0]
        assert req.request.limit == 2

    @pytest.mark.asyncio
    async def test_created_after_filter(self):
        client = MagicMock()
        client.artifact_service.list_artifacts = AsyncMock(
            return_value=artifact_service_pb2.ListArtifactsResponse(artifacts=[], token="")
        )

        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            _ = [
                a
                async for a in Artifact.listall.aio(
                    created_after=datetime(2026, 1, 1, tzinfo=timezone.utc),
                )
            ]

        req = client.artifact_service.list_artifacts.await_args[0][0]
        assert len(req.request.filters) == 1
        f = req.request.filters[0]
        assert f.field == "created_at"
        assert f.function == list_pb2.Filter.GREATER_THAN
        assert f.values == ["2026-01-01T00:00:00Z"]

    @pytest.mark.asyncio
    async def test_source_filters(self):
        client = MagicMock()
        client.artifact_service.list_artifacts = AsyncMock(
            return_value=artifact_service_pb2.ListArtifactsResponse(artifacts=[], token="")
        )

        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            _ = [a async for a in Artifact.listall.aio(source_run="r1", source_action="a0")]

        req = client.artifact_service.list_artifacts.await_args[0][0]
        got = {f.field: (f.function, list(f.values)) for f in req.request.filters}
        assert got == {
            "source_run": (list_pb2.Filter.EQUAL, ["r1"]),
            "source_action": (list_pb2.Filter.EQUAL, ["a0"]),
        }

    @pytest.mark.asyncio
    async def test_source_external_ref_filter(self):
        client = MagicMock()
        client.artifact_service.list_artifacts = AsyncMock(
            return_value=artifact_service_pb2.ListArtifactsResponse(artifacts=[], token="")
        )

        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            _ = [a async for a in Artifact.listall.aio(source_external_ref="hf://org/model")]

        req = client.artifact_service.list_artifacts.await_args[0][0]
        assert len(req.request.filters) == 1
        f = req.request.filters[0]
        assert f.field == "source_external_ref"
        assert f.function == list_pb2.Filter.EQUAL
        assert f.values == ["hf://org/model"]


class TestToPython:
    @pytest.mark.asyncio
    async def test_round_trip_guesses_type(self):
        artifact = Artifact(pb2=await _stored_artifact("round-trip"))
        assert await artifact.to_python() == "round-trip"


class TestCoerceToLiteral:
    """coerce_to_literal round-trips the stored literal through the type engine against the
    declared type, so compatibility rules are the engine's, and stamps the artifact's identity
    on the result."""

    async def _file_artifact(self, uri: str = "s3://bucket/weights.pt") -> Artifact:
        lt = TypeEngine.to_literal_type(File)
        lit = await TypeEngine.to_literal(File(path=uri), File, lt)
        return Artifact(
            pb2=artifact_pb2.Artifact(
                artifact_id=artifact_pb2.ArtifactIdentifier(
                    name=artifact_pb2.ArtifactName(org="test-org", project="proj", domain="dev", name="m"),
                    version="v1",
                ),
                spec=artifact_pb2.ArtifactSpec(value=lit, type=lt),
            )
        )

    @pytest.mark.asyncio
    async def test_no_type_returns_stored_literal(self):
        artifact = Artifact(pb2=await _stored_artifact("as-is"))
        assert await artifact.coerce_to_literal() is artifact.pb2.spec.value

    @pytest.mark.asyncio
    async def test_coerces_and_stamps_identity(self):
        artifact = await self._file_artifact()
        lit = await artifact.coerce_to_literal(File)
        assert lit.scalar.blob.uri == "s3://bucket/weights.pt"
        assert lit.artifact_id == artifact.artifact_version_id

    @pytest.mark.asyncio
    async def test_optional_type_wraps_in_union(self):
        from typing import Optional

        artifact = await self._file_artifact()
        lit = await artifact.coerce_to_literal(Optional[File])
        assert lit.scalar.WhichOneof("value") == "union"
        assert lit.scalar.union.value.scalar.blob.uri == "s3://bucket/weights.pt"
        assert lit.artifact_id == artifact.artifact_version_id

    @pytest.mark.asyncio
    async def test_mismatch_raises_transformer_error(self):
        from flyte.types import TypeTransformerFailedError

        artifact = await self._file_artifact()
        with pytest.raises(TypeTransformerFailedError):
            await artifact.coerce_to_literal(str)

    @pytest.mark.asyncio
    async def test_prefers_stored_stamp_over_record_id(self):
        """When the stored literal already carries an artifact_id (the normal service case),
        that stamp wins over the record's version id."""
        from flyteidl2.core import artifact_id_pb2

        artifact = await self._file_artifact()
        stamped = artifact_id_pb2.ArtifactVersionId(
            key=artifact_id_pb2.ArtifactKey(org="test-org", project="proj", domain="dev", name="m"),
            version="stamped-by-service",
        )
        artifact.pb2.spec.value.artifact_id.CopyFrom(stamped)
        lit = await artifact.coerce_to_literal(File)
        assert lit.artifact_id.version == "stamped-by-service"


class TestListNames:
    @pytest.mark.asyncio
    async def test_groups_paginate_and_wrap(self):
        stored = await _stored_artifact("v", name="model-a", version="v3")
        group = artifact_service_pb2.ArtifactGroup(latest=stored, versions=3)
        client = MagicMock()
        client.artifact_service.list_artifact_names = AsyncMock(
            side_effect=[
                artifact_service_pb2.ListArtifactNamesResponse(groups=[group], token="1"),
                artifact_service_pb2.ListArtifactNamesResponse(groups=[group], token=""),
            ]
        )

        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            groups = [g async for g in Artifact.list_names.aio()]

        assert len(groups) == 2
        assert groups[0].name == "model-a"
        assert groups[0].versions == 3
        assert groups[0].latest.version == "v3"
        assert client.artifact_service.list_artifact_names.await_count == 2

    @pytest.mark.asyncio
    async def test_search_filter(self):
        client = MagicMock()
        client.artifact_service.list_artifact_names = AsyncMock(
            return_value=artifact_service_pb2.ListArtifactNamesResponse(groups=[], token="")
        )

        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            _ = [g async for g in Artifact.list_names.aio(search="model")]

        req = client.artifact_service.list_artifact_names.await_args[0][0]
        assert len(req.request.filters) == 1
        assert req.request.filters[0].field == "name"
        assert req.request.filters[0].function == list_pb2.Filter.CONTAINS
        assert req.request.filters[0].values == ["model"]

    @pytest.mark.asyncio
    async def test_limit_stops_early(self):
        stored = await _stored_artifact("v", name="m")
        groups_page = [artifact_service_pb2.ArtifactGroup(latest=stored, versions=1) for _ in range(3)]
        client = MagicMock()
        client.artifact_service.list_artifact_names = AsyncMock(
            return_value=artifact_service_pb2.ListArtifactNamesResponse(groups=groups_page, token="t")
        )

        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            got = [g async for g in Artifact.list_names.aio(limit=2)]

        assert len(got) == 2
