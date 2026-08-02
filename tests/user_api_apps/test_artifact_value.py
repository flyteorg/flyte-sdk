"""Tests for flyte.app.ArtifactValue: artifacts as app parameter values."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import flyte.errors
from flyte.app import ArtifactValue, Parameter
from flyte.app._parameter import SerializableParameter
from flyte.io import Dir, File


def test_artifact_value_minimal():
    av = ArtifactValue(type="directory", name="bert-small")
    assert av.name == "bert-small"
    assert av.version is None
    assert av.project is None
    assert av.domain is None


def test_artifact_value_pinned_version():
    av = ArtifactValue(type="file", name="weights", version="abc123", project="p", domain="d")
    assert av.version == "abc123"
    assert av.project == "p"
    assert av.domain == "d"


def test_artifact_value_python_type_mapping():
    assert ArtifactValue(type=File, name="a").type == "file"
    assert ArtifactValue(type=Dir, name="a").type == "directory"


def test_artifact_value_json_roundtrip():
    av = ArtifactValue(type="directory", name="bert-small", version="v1")
    restored = ArtifactValue.model_validate_json(av.model_dump_json())
    assert restored == av


def test_parameter_accepts_artifact_value():
    p = Parameter(name="model", value=ArtifactValue(type="directory", name="bert-small"), mount="/models")
    assert isinstance(p.value, ArtifactValue)


def test_serializable_parameter_from_unmaterialized_artifact_value():
    p = Parameter(name="model", value=ArtifactValue(type="directory", name="bert-small"), mount="/models")
    sp = SerializableParameter.from_parameter(p)
    assert sp.type == "directory"
    assert sp.download is True
    assert '"name":"bert-small"' in sp.value


def _patched_remote_artifact(value):
    artifact = MagicMock()
    artifact.version = "v1"
    artifact.to_python = AsyncMock(return_value=value)
    get = AsyncMock(return_value=artifact)
    mock_artifact_cls = MagicMock()
    mock_artifact_cls.get.aio = get
    return mock_artifact_cls, get


@pytest.mark.asyncio
async def test_materialize_directory_artifact():
    mock_cls, get = _patched_remote_artifact(Dir(path="s3://bucket/models/bert"))
    with patch("flyte._initialize.is_initialized", return_value=True), patch("flyte.remote.Artifact", mock_cls):
        value = await ArtifactValue(type="directory", name="bert-small").materialize()

    assert isinstance(value, Dir)
    assert value.path == "s3://bucket/models/bert"
    get.assert_awaited_once_with("bert-small", version="latest", project=None, domain=None)


@pytest.mark.asyncio
async def test_materialize_pinned_version_and_scope():
    mock_cls, get = _patched_remote_artifact(File(path="s3://bucket/weights.pt"))
    with patch("flyte._initialize.is_initialized", return_value=True), patch("flyte.remote.Artifact", mock_cls):
        value = await ArtifactValue(type="file", name="weights", version="abc", project="p", domain="d").materialize()

    assert isinstance(value, File)
    get.assert_awaited_once_with("weights", version="abc", project="p", domain="d")


@pytest.mark.asyncio
async def test_materialize_type_mismatch_raises():
    mock_cls, _ = _patched_remote_artifact(File(path="s3://bucket/weights.pt"))
    with (
        patch("flyte._initialize.is_initialized", return_value=True),
        patch("flyte.remote.Artifact", mock_cls),
        pytest.raises(flyte.errors.ParameterMaterializationError, match="declared as 'directory'"),
    ):
        await ArtifactValue(type="directory", name="weights").materialize()


@pytest.mark.asyncio
async def test_materialize_string_type_rejected():
    with patch("flyte._initialize.is_initialized", return_value=True), pytest.raises(ValueError, match="file"):
        await ArtifactValue(type="string", name="weights").materialize()


@pytest.mark.asyncio
async def test_materialize_lookup_failure_wrapped():
    mock_cls = MagicMock()
    mock_cls.get.aio = AsyncMock(side_effect=RuntimeError("not found"))
    with (
        patch("flyte._initialize.is_initialized", return_value=True),
        patch("flyte.remote.Artifact", mock_cls),
        pytest.raises(flyte.errors.ParameterMaterializationError, match="weights@latest"),
    ):
        await ArtifactValue(type="file", name="weights").materialize()
