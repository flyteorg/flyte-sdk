"""Tests for flyte.remote.Artifact create/get/listall against a fake ArtifactService."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.artifact import artifact_pb2, artifact_service_pb2
from flyteidl2.common import list_pb2

import flyte.artifacts as artifacts
from flyte.remote._artifact import Artifact
from flyte.types import TypeEngine


def _cfg():
    return MagicMock(org="test-org", project="proj", domain="dev")


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
            result = await Artifact.create.aio("hello", name="my_artifact", version="1.0", description="desc")

        req = client.artifact_service.create_artifact.await_args[0][0]
        assert req.artifact_id.name.org == "test-org"
        assert req.artifact_id.name.project == "proj"
        assert req.artifact_id.name.domain == "dev"
        assert req.artifact_id.name.name == "my_artifact"
        assert req.artifact_id.version == "1.0"
        assert req.spec.description == "desc"
        # The value round-trips through the type engine as a string literal.
        assert await TypeEngine.to_python_value(req.spec.value, str) == "hello"
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
            data={"framework": "torch"},
            card=artifacts.Card(uri="s3://b/card.html", format="html", card_type="model"),
        )

        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            await Artifact.create.aio(artifacts.new("v", md))

        req = client.artifact_service.create_artifact.await_args[0][0]
        assert req.artifact_id.name.name == "wrapped"
        assert req.artifact_id.version == "2.0"
        assert req.spec.description == "from metadata"
        assert dict(req.spec.user_metadata) == {"framework": "torch"}
        assert req.spec.card.uri == "s3://b/card.html"
        assert req.spec.card.type == "model"

    @pytest.mark.asyncio
    async def test_create_defaults_random_version(self):
        client = MagicMock()
        client.artifact_service.create_artifact = AsyncMock(
            return_value=artifact_service_pb2.CreateArtifactResponse(artifact=await _stored_artifact("v"))
        )

        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            await Artifact.create.aio("v", name="unversioned")

        req = client.artifact_service.create_artifact.await_args[0][0]
        assert req.artifact_id.version  # non-empty random version

    @pytest.mark.asyncio
    async def test_create_without_name_raises(self):
        client = MagicMock()
        p1, p2, p3 = _patched(client)
        with p1, p2, p3, pytest.raises(ValueError, match="name is required"):
            await Artifact.create.aio("v")


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


class TestToPython:
    @pytest.mark.asyncio
    async def test_round_trip_guesses_type(self):
        artifact = Artifact(pb2=await _stored_artifact("round-trip"))
        assert await artifact.to_python() == "round-trip"
