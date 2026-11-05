"""Tests for Artifact.download_card: a signed link from the data proxy, with a direct storage fallback."""

from __future__ import annotations

import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.artifact import artifact_pb2
from flyteidl2.core import artifact_id_pb2

from flyte.remote._artifact import Artifact
from flyte.remote._artifact_card import card_link_request, supports_card_links

CARD_URI = "s3://bucket/runs/r1/a0/cards/model-0123abcd.html"
MODULE = "flyte.remote._artifact_card"


def _artifact(card_uri: str = CARD_URI) -> Artifact:
    info = artifact_id_pb2.ArtifactInfo()
    if card_uri:
        info.card.CopyFrom(artifact_id_pb2.ArtifactCard(uri=card_uri, format="html", type="model"))
    return Artifact(
        pb2=artifact_pb2.Artifact(
            artifact_id=artifact_pb2.ArtifactIdentifier(
                name=artifact_pb2.ArtifactName(org="o", project="p", domain="d", name="trained-model"),
                version="v1",
            ),
            spec=artifact_pb2.ArtifactSpec(info=info),
        )
    )


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


def _client(signed_urls: list[str]) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.pre_signed_urls.signed_url = signed_urls
    client.dataproxy_service.create_download_link = AsyncMock(return_value=response)
    return client


def test_download_card_via_signed_link(tmp_path: pathlib.Path):
    client = _client(["https://signed/card"])
    storage_get = AsyncMock()
    with (
        patch(f"{MODULE}.supports_card_links", return_value=True),
        patch(f"{MODULE}.card_link_request", return_value="card-request") as request,
        patch(f"{MODULE}.ensure_client"),
        patch(f"{MODULE}.get_client", return_value=client),
        _patch_httpx({"https://signed/card": b"<h1>model</h1>"}),
        patch("flyte.storage.get", storage_get),
    ):
        path = _artifact().download_card(tmp_path)

    assert path == tmp_path / "model-0123abcd.html"
    assert path.read_bytes() == b"<h1>model</h1>"
    request.assert_called_once()
    client.dataproxy_service.create_download_link.assert_awaited_once_with("card-request")
    storage_get.assert_not_called()


def test_download_card_to_a_file_path(tmp_path: pathlib.Path):
    client = _client(["https://signed/card"])
    dest = tmp_path / "cards" / "model.html"
    with (
        patch(f"{MODULE}.supports_card_links", return_value=True),
        patch(f"{MODULE}.card_link_request"),
        patch(f"{MODULE}.ensure_client"),
        patch(f"{MODULE}.get_client", return_value=client),
        _patch_httpx({"https://signed/card": b"card"}),
    ):
        path = _artifact().download_card(dest)

    assert path == dest
    assert dest.read_bytes() == b"card"


def test_default_destination_is_the_cwd(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    storage_get = AsyncMock()
    with patch(f"{MODULE}.supports_card_links", return_value=False), patch("flyte.storage.get", storage_get):
        path = _artifact().download_card()

    assert path == tmp_path / "model-0123abcd.html"
    storage_get.assert_awaited_once_with(CARD_URI, str(path))


def test_falls_back_to_storage_when_the_backend_cannot_sign(tmp_path: pathlib.Path):
    client = MagicMock()
    client.dataproxy_service.create_download_link = AsyncMock(side_effect=RuntimeError("unsupported artifact type"))
    storage_get = AsyncMock()
    with (
        patch(f"{MODULE}.supports_card_links", return_value=True),
        patch(f"{MODULE}.card_link_request"),
        patch(f"{MODULE}.ensure_client"),
        patch(f"{MODULE}.get_client", return_value=client),
        patch("flyte.storage.get", storage_get),
    ):
        path = _artifact().download_card(tmp_path / "card.html")

    assert path == tmp_path / "card.html"
    storage_get.assert_awaited_once_with(CARD_URI, str(tmp_path / "card.html"))


def test_without_card_link_support_the_proxy_is_not_called(tmp_path: pathlib.Path):
    """A flyteidl2 build without artifact card links goes straight to storage."""
    client = _client(["https://signed/card"])
    storage_get = AsyncMock()
    with (
        patch(f"{MODULE}.supports_card_links", return_value=False),
        patch(f"{MODULE}.ensure_client"),
        patch(f"{MODULE}.get_client", return_value=client),
        patch("flyte.storage.get", storage_get),
    ):
        _artifact().download_card(tmp_path / "card.html")

    client.dataproxy_service.create_download_link.assert_not_called()
    storage_get.assert_awaited_once()


def test_reports_both_errors_when_neither_route_works(tmp_path: pathlib.Path):
    client = MagicMock()
    client.dataproxy_service.create_download_link = AsyncMock(side_effect=RuntimeError("signing unsupported"))
    storage_get = AsyncMock(side_effect=OSError("Generic S3 error: 169.254.169.254"))
    with (
        patch(f"{MODULE}.supports_card_links", return_value=True),
        patch(f"{MODULE}.card_link_request"),
        patch(f"{MODULE}.ensure_client"),
        patch(f"{MODULE}.get_client", return_value=client),
        patch("flyte.storage.get", storage_get),
    ):
        with pytest.raises(RuntimeError) as err:
            _artifact().download_card(tmp_path / "card.html")

    assert "signing unsupported" in str(err.value)
    assert "169.254.169.254" in str(err.value)
    assert not (tmp_path / "card.html").exists()


def test_an_artifact_without_a_card_is_rejected(tmp_path: pathlib.Path):
    with pytest.raises(ValueError, match="has no card"):
        _artifact(card_uri="").download_card(tmp_path)


@pytest.mark.skipif(not supports_card_links(), reason="installed flyteidl2 predates artifact card download links")
def test_card_link_request_names_the_artifact_version():
    from flyteidl2.dataproxy import dataproxy_service_pb2

    art = _artifact()
    request = card_link_request(art)

    assert request.artifact_type == dataproxy_service_pb2.ARTIFACT_TYPE_CARD
    assert request.WhichOneof("source") == "artifact_id"
    assert request.artifact_id == art.pb2.artifact_id
