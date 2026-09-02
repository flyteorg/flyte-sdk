"""Tests for RemoteImageChecker.image_exists and its build-run recording side channel."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from flyteidl2.common import identifier_pb2
from flyteidl2.imagebuilder import definition_pb2 as image_definition_pb2

import flyte._internal.imagebuild.image_builder as ib
from flyte._internal.imagebuild.remote_builder import RemoteImageChecker, _maybe_record_build_run


@pytest.fixture(autouse=True)
def _fresh_build_run_registry(monkeypatch):
    monkeypatch.setattr(ib, "_image_build_runs", {})


async def _check_image(image_pb: image_definition_pb2.Image):
    """Run RemoteImageChecker.image_exists with the backend mocked to return image_pb."""
    from flyteidl2.imagebuilder import payload_pb2 as image_payload_pb2

    cfg = MagicMock()
    cfg.org = "testorg"
    cfg.project = "flytesnacks"
    cfg.domain = "development"
    cfg.client.image_service.get_image = AsyncMock(return_value=image_payload_pb2.GetImageResponse(image=image_pb))
    with (
        patch("flyte.remote.Task.get"),
        patch("flyte._initialize._get_init_config", return_value=cfg),
    ):
        return await RemoteImageChecker.image_exists("registry.example.com/my-image", "v1.0")


@pytest.mark.asyncio
async def test_records_build_run_on_hit():
    image_pb = image_definition_pb2.Image(
        id=image_definition_pb2.ImageIdentifier(name="my-image:v1.0"),
        fqin="registry.example.com/my-image:v1.0",
        build_run=identifier_pb2.RunIdentifier(
            org="testorg", project="flytesnacks", domain="development", name="run123"
        ),
    )
    result = await _check_image(image_pb)
    assert result == "registry.example.com/my-image:v1.0"
    assert ib.get_image_build_run("registry.example.com/my-image:v1.0") == ib.RunIdentifierData(
        org="testorg", project="flytesnacks", domain="development", name="run123"
    )


@pytest.mark.asyncio
async def test_unset_build_run_records_nothing():
    """Old servers leave build_run unset — the hit still resolves, no run is recorded."""
    image_pb = image_definition_pb2.Image(fqin="registry.example.com/my-image:v1.0")
    result = await _check_image(image_pb)
    assert result == "registry.example.com/my-image:v1.0"
    assert ib.get_image_build_run("registry.example.com/my-image:v1.0") is None


def test_partial_build_run_records_nothing():
    """The console needs domain, project and name; a partial identifier is not recorded."""
    image_pb = image_definition_pb2.Image(
        fqin="registry.example.com/my-image:v1.0",
        build_run=identifier_pb2.RunIdentifier(name="run123"),
    )
    _maybe_record_build_run(image_pb)
    assert ib.get_image_build_run("registry.example.com/my-image:v1.0") is None


def test_recording_failure_is_swallowed():
    """A broken message can never propagate out of the recording helper."""
    _maybe_record_build_run(Mock())
