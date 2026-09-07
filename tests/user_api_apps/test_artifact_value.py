"""Tests for flyte.app.ArtifactValue: artifacts as app parameter values."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import flyte.errors
from flyte.app import ArtifactValue, Parameter
from flyte.app._parameter import SerializableParameter
from flyte.io import Dir, File


def test_artifact_value_minimal():
    av = ArtifactValue(name="bert-small")
    assert av.name == "bert-small"
    assert av.version is None
    assert av.project is None
    assert av.domain is None
    assert av.type is None  # inferred from the artifact at materialization


def test_artifact_value_pinned_version():
    av = ArtifactValue(name="weights", version="abc123", project="p", domain="d")
    assert av.version == "abc123"
    assert av.project == "p"
    assert av.domain == "d"


def test_artifact_value_python_type_mapping():
    assert ArtifactValue(type=File, name="a").type == "file"
    assert ArtifactValue(type=Dir, name="a").type == "directory"


def test_artifact_value_json_roundtrip():
    av = ArtifactValue(name="bert-small", version="v1")
    restored = ArtifactValue.model_validate_json(av.model_dump_json())
    assert restored == av


def test_parameter_accepts_artifact_value():
    p = Parameter(name="model", value=ArtifactValue(name="bert-small"), mount="/models")
    assert isinstance(p.value, ArtifactValue)


def test_serializable_parameter_from_unmaterialized_declared_type():
    p = Parameter(name="model", value=ArtifactValue(type="directory", name="bert-small"), mount="/models")
    sp = SerializableParameter.from_parameter(p)
    assert sp.type == "directory"
    assert sp.download is True
    assert '"name":"bert-small"' in sp.value


def test_serializable_parameter_from_unmaterialized_inferred_type_raises():
    p = Parameter(name="model", value=ArtifactValue(name="bert-small"), mount="/models")
    with pytest.raises(ValueError, match="materialized"):
        SerializableParameter.from_parameter(p)


def _patched_remote_artifact(value):
    artifact = MagicMock()
    artifact.version = "v1"
    artifact.to_python = AsyncMock(return_value=value)
    get = AsyncMock(return_value=artifact)
    mock_artifact_cls = MagicMock()
    mock_artifact_cls.get.aio = get
    return mock_artifact_cls, get


@pytest.mark.asyncio
async def test_materialize_infers_directory():
    mock_cls, get = _patched_remote_artifact(Dir(path="s3://bucket/models/bert"))
    with patch("flyte._initialize.is_initialized", return_value=True), patch("flyte.remote.Artifact", mock_cls):
        value = await ArtifactValue(name="bert-small").materialize()

    assert isinstance(value, Dir)
    assert value.path == "s3://bucket/models/bert"
    get.assert_awaited_once_with("bert-small", version="latest", project=None, domain=None)


@pytest.mark.asyncio
async def test_materialize_infers_file():
    mock_cls, _ = _patched_remote_artifact(File(path="s3://bucket/weights.pt"))
    with patch("flyte._initialize.is_initialized", return_value=True), patch("flyte.remote.Artifact", mock_cls):
        value = await ArtifactValue(name="weights").materialize()

    assert isinstance(value, File)


@pytest.mark.asyncio
async def test_materialize_pinned_version_and_scope():
    mock_cls, get = _patched_remote_artifact(File(path="s3://bucket/weights.pt"))
    with patch("flyte._initialize.is_initialized", return_value=True), patch("flyte.remote.Artifact", mock_cls):
        value = await ArtifactValue(type="file", name="weights", version="abc", project="p", domain="d").materialize()

    assert isinstance(value, File)
    get.assert_awaited_once_with("weights", version="abc", project="p", domain="d")


@pytest.mark.asyncio
async def test_materialize_declared_type_mismatch_raises():
    mock_cls, _ = _patched_remote_artifact(File(path="s3://bucket/weights.pt"))
    with (
        patch("flyte._initialize.is_initialized", return_value=True),
        patch("flyte.remote.Artifact", mock_cls),
        pytest.raises(flyte.errors.ParameterMaterializationError, match="declared as 'directory'"),
    ):
        await ArtifactValue(type="directory", name="weights").materialize()


@pytest.mark.asyncio
async def test_materialize_non_asset_artifact_raises():
    mock_cls, _ = _patched_remote_artifact("a string value")
    with (
        patch("flyte._initialize.is_initialized", return_value=True),
        patch("flyte.remote.Artifact", mock_cls),
        pytest.raises(flyte.errors.ParameterMaterializationError, match="only File and Dir"),
    ):
        await ArtifactValue(name="weights").materialize()


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


@pytest.mark.asyncio
async def test_materialize_records_resolved_version_id():
    """Materialization keeps the artifact identity that the value itself loses."""
    mock_cls, _ = _patched_remote_artifact(File(path="s3://bucket/weights.pt"))
    av = ArtifactValue(name="weights")
    assert av.resolved_version_id is None

    with patch("flyte._initialize.is_initialized", return_value=True), patch("flyte.remote.Artifact", mock_cls):
        await av.materialize()

    # The mock's artifact_version_id stands in for the real ArtifactVersionId.
    assert av.resolved_version_id is not None
    assert av.resolved_version_id is mock_cls.get.aio.return_value.artifact_version_id
