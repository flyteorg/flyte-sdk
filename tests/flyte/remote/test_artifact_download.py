"""Tests for Artifact.download: signed links from the data proxy, with a direct storage fallback."""

from __future__ import annotations

import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.artifact import artifact_pb2
from flyteidl2.core import literals_pb2, types_pb2

from flyte.remote._artifact import Artifact
from flyte.remote._artifact_download import _supports_artifact_links, value_link_request

FILE_URI = "s3://bucket/runs/r1/a0/policy.json"
DIR_URI = "s3://bucket/runs/r1/a0/scored-rollouts"
MODULE = "flyte.remote._artifact_download"


def _artifact(value: literals_pb2.Literal, name: str = "rollouts", version: str = "r1-a0-1") -> Artifact:
    return Artifact(
        pb2=artifact_pb2.Artifact(
            artifact_id=artifact_pb2.ArtifactIdentifier(
                name=artifact_pb2.ArtifactName(org="o", project="p", domain="d", name=name),
                version=version,
            ),
            spec=artifact_pb2.ArtifactSpec(value=value),
        )
    )


def _blob_artifact(uri: str, dimensionality: int) -> Artifact:
    blob = literals_pb2.Blob(
        uri=uri, metadata=literals_pb2.BlobMetadata(type=types_pb2.BlobType(dimensionality=dimensionality))
    )
    return _artifact(literals_pb2.Literal(scalar=literals_pb2.Scalar(blob=blob)))


class _FakeStreamResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self):
        yield self._payload


def _patch_httpx(payloads: dict[str, bytes]):
    """Patch httpx.AsyncClient so stream(GET, url) yields payloads[url] without network."""

    def stream(_method: str, url: str):
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=_FakeStreamResponse(payloads[url]))
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    client = MagicMock()
    client.stream = MagicMock(side_effect=stream)
    client_ctx = MagicMock()
    client_ctx.__aenter__ = AsyncMock(return_value=client)
    client_ctx.__aexit__ = AsyncMock(return_value=False)
    return patch(f"{MODULE}.httpx.AsyncClient", return_value=client_ctx)


def test_download_file_via_signed_link(tmp_path: pathlib.Path):
    art = _blob_artifact(FILE_URI, types_pb2.BlobType.SINGLE)
    links = AsyncMock(return_value=(["https://signed/policy"], []))
    storage_get = AsyncMock()
    with (
        patch(f"{MODULE}._signed_links", links),
        _patch_httpx({"https://signed/policy": b'{"w": 1}'}),
        patch("flyte.storage.get", storage_get),
    ):
        path = art.download(tmp_path)

    assert path == tmp_path / "policy.json"
    assert path.read_bytes() == b'{"w": 1}'
    storage_get.assert_not_called()


def test_download_directory_keeps_relative_paths(tmp_path: pathlib.Path):
    art = _blob_artifact(DIR_URI, types_pb2.BlobType.MULTIPART)
    links = AsyncMock(return_value=(["https://signed/a", "https://signed/b"], ["a.jsonl", "sub/b.jsonl"]))
    dest = tmp_path / "rollouts"
    with (
        patch(f"{MODULE}._signed_links", links),
        _patch_httpx({"https://signed/a": b"a", "https://signed/b": b"b"}),
    ):
        path = art.download(dest)

    assert path == dest
    assert (dest / "a.jsonl").read_bytes() == b"a"
    assert (dest / "sub" / "b.jsonl").read_bytes() == b"b"


def test_download_directory_refuses_paths_outside_destination(tmp_path: pathlib.Path):
    art = _blob_artifact(DIR_URI, types_pb2.BlobType.MULTIPART)
    links = AsyncMock(return_value=(["https://signed/a"], ["../escape.txt"]))
    storage_get = AsyncMock(side_effect=PermissionError("no credentials"))
    with patch(f"{MODULE}._signed_links", links), patch("flyte.storage.get", storage_get):
        with pytest.raises(RuntimeError, match="unsafe relative path"):
            art.download(tmp_path / "rollouts")
    assert not (tmp_path / "escape.txt").exists()


def test_falls_back_to_storage_when_the_backend_cannot_sign(tmp_path: pathlib.Path):
    """An older backend rejects ARTIFACT_TYPE_VALUE; reading the store directly still works in-cluster."""
    art = _blob_artifact(DIR_URI, types_pb2.BlobType.MULTIPART)
    links = AsyncMock(side_effect=RuntimeError("artifact_id source requires artifact type ARTIFACT_TYPE_CARD"))
    storage_get = AsyncMock(return_value=str(tmp_path / "rollouts"))
    with patch(f"{MODULE}._signed_links", links), patch("flyte.storage.get", storage_get):
        path = art.download(tmp_path / "rollouts")

    assert path == tmp_path / "rollouts"
    storage_get.assert_awaited_once_with(DIR_URI, str(tmp_path / "rollouts"), recursive=True)


def test_reports_both_errors_when_neither_path_works(tmp_path: pathlib.Path):
    art = _blob_artifact(FILE_URI, types_pb2.BlobType.SINGLE)
    links = AsyncMock(side_effect=RuntimeError("signing unsupported"))
    storage_get = AsyncMock(side_effect=OSError("Generic S3 error: 169.254.169.254"))
    with patch(f"{MODULE}._signed_links", links), patch("flyte.storage.get", storage_get):
        with pytest.raises(RuntimeError) as err:
            art.download(tmp_path / "policy.json")

    message = str(err.value)
    assert "signing unsupported" in message
    assert "169.254.169.254" in message
    assert "storage credentials" in message


def test_non_blob_value_is_rejected():
    art = _artifact(literals_pb2.Literal(scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(integer=3))))
    with pytest.raises(ValueError, match="to_python"):
        art.download("/tmp/unused")


def test_default_destination_is_the_blob_name_under_cwd(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    art = _blob_artifact(DIR_URI + "/", types_pb2.BlobType.MULTIPART)
    links = AsyncMock(return_value=([], []))
    with patch(f"{MODULE}._signed_links", links), _patch_httpx({}):
        path = art.download()

    assert path == tmp_path / "scored-rollouts"
    assert path.is_dir()


def test_without_artifact_link_support_the_proxy_is_not_called(tmp_path: pathlib.Path):
    """A flyteidl2 build that predates artifact download links goes straight to storage."""
    art = _blob_artifact(FILE_URI, types_pb2.BlobType.SINGLE)
    client = MagicMock()
    client.dataproxy_service.create_download_link = AsyncMock()
    storage_get = AsyncMock(return_value=str(tmp_path / "policy.json"))
    with (
        patch(f"{MODULE}._supports_artifact_links", return_value=False),
        patch(f"{MODULE}.ensure_client"),
        patch(f"{MODULE}.get_client", return_value=client),
        patch("flyte.storage.get", storage_get),
    ):
        art.download(tmp_path / "policy.json")

    client.dataproxy_service.create_download_link.assert_not_called()
    storage_get.assert_awaited_once()


@pytest.mark.skipif(not _supports_artifact_links(), reason="installed flyteidl2 predates artifact download links")
def test_value_link_request_names_the_artifact_version():
    art = _blob_artifact(DIR_URI, types_pb2.BlobType.MULTIPART)
    request = value_link_request(art)

    from flyteidl2.dataproxy import dataproxy_service_pb2

    assert request.artifact_type == dataproxy_service_pb2.ARTIFACT_TYPE_VALUE
    assert request.WhichOneof("source") == "artifact_id"
    assert request.artifact_id == art.pb2.artifact_id
