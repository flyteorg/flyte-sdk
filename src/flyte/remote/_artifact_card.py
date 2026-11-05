"""
Download an artifact version's card (the model or data card document) to the local machine.

Two routes to the bytes, tried in this order:

1. The data proxy. `CreateDownloadLink` with the artifact's id and `ARTIFACT_TYPE_CARD` returns a
   pre-signed URL, so no object store credentials are needed on the client. flyteidl2 builds
   without these fields, and backends that don't sign artifact cards, fail here and fall through.
2. A direct object store read of the card's URI, via `flyte.storage`. This works wherever the
   caller has credentials for the bucket, such as inside a task.

If both fail, the error names both causes.
"""

from __future__ import annotations

import pathlib
import posixpath
from typing import TYPE_CHECKING

import httpx
from flyteidl2.dataproxy import dataproxy_service_pb2

from flyte._initialize import ensure_client, get_client
from flyte._logging import logger

if TYPE_CHECKING:
    from flyte.remote._artifact import Artifact

_DOWNLOAD_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=10.0)


def _card_uri(artifact: Artifact) -> str:
    uri = artifact.pb2.spec.info.card.uri
    if not uri:
        raise ValueError(f"Artifact {artifact.name}@{artifact.version} has no card.")
    return uri


def card_file_name(artifact: Artifact) -> str:
    """The file name a card is written to when no file path is given: the card object's own name."""
    name = posixpath.basename(_card_uri(artifact).rstrip("/"))
    if name:
        return name
    card_format = artifact.pb2.spec.info.card.format or "html"
    return f"{artifact.name}.{card_format}"


def supports_card_links() -> bool:
    """Whether the installed flyteidl2 can ask the data proxy for an artifact's card."""
    fields = dataproxy_service_pb2.CreateDownloadLinkRequest.DESCRIPTOR.fields_by_name  # ty: ignore[unresolved-attribute]
    return "artifact_id" in fields and hasattr(dataproxy_service_pb2, "ARTIFACT_TYPE_CARD")


def card_link_request(artifact: Artifact) -> dataproxy_service_pb2.CreateDownloadLinkRequest:
    """The CreateDownloadLink request that signs `artifact`'s card."""
    request = dataproxy_service_pb2.CreateDownloadLinkRequest(
        artifact_type=dataproxy_service_pb2.ARTIFACT_TYPE_CARD,  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    )
    request.artifact_id.CopyFrom(artifact.pb2.artifact_id)  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    return request


async def _download_via_data_proxy(artifact: Artifact, target: pathlib.Path) -> None:
    if not supports_card_links():
        raise RuntimeError("the installed flyteidl2 can't request download links for artifact cards")
    ensure_client()
    resp = await get_client().dataproxy_service.create_download_link(card_link_request(artifact))
    signed_urls = list(resp.pre_signed_urls.signed_url)
    if not signed_urls:
        raise RuntimeError("the data proxy returned no download link for the card")

    async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
        async with client.stream("GET", signed_urls[0]) as response:
            response.raise_for_status()
            with target.open("wb") as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)


async def _download_via_storage(uri: str, target: pathlib.Path) -> None:
    import flyte.storage as storage

    await storage.get(uri, str(target))


async def download_card(artifact: Artifact, dest: pathlib.Path) -> pathlib.Path:
    """
    Download `artifact`'s card to `dest` and return the local path.

    The card is written to `dest`, or inside it under the card's own name when `dest` is an
    existing directory.
    """
    uri = _card_uri(artifact)
    if dest.is_dir():
        dest = dest / card_file_name(artifact)
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        await _download_via_data_proxy(artifact, dest)
    except Exception as proxy_error:
        logger.debug(
            f"Could not download the card of artifact {artifact.name}@{artifact.version} through the data proxy "
            f"({proxy_error}), reading {uri} directly."
        )
        try:
            await _download_via_storage(uri, dest)
        except Exception as storage_error:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download the card of artifact {artifact.name}@{artifact.version}. The data proxy "
                f"could not sign it ({proxy_error}), and reading {uri} directly failed ({storage_error})."
            ) from storage_error
    return dest
