"""
Download the data behind an artifact version (a file or a directory) to the local machine.

The data lives in the tenant's object store, which a laptop usually has no credentials for. So
the download goes through the data proxy first: `CreateDownloadLink` with the artifact's id and
`ARTIFACT_TYPE_VALUE` returns pre-signed URLs, one for a file and one per object for a directory
(with each object's path under the directory), and the objects are fetched over HTTPS.

Older backends, and flyteidl2 builds without these fields, can't sign artifact data. The download
then reads the object store directly, which works where storage credentials are available (inside
a task, or with credentials configured locally), and otherwise fails with both errors.
"""

from __future__ import annotations

import pathlib
import posixpath
from typing import TYPE_CHECKING

import httpx
from flyteidl2.core import types_pb2
from flyteidl2.dataproxy import dataproxy_service_pb2

from flyte._initialize import ensure_client, get_client
from flyte._logging import logger

if TYPE_CHECKING:
    from flyteidl2.core import literals_pb2

    from flyte.remote._artifact import Artifact

_DOWNLOAD_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=10.0)


class ArtifactDownloadNotSupported(RuntimeError):
    """The installed flyteidl2 can't ask the data proxy for an artifact's data."""


def _blob(artifact: Artifact) -> literals_pb2.Blob:
    value = artifact.pb2.spec.value
    if not (value.HasField("scalar") and value.scalar.HasField("blob")):
        raise ValueError(
            f"Artifact {artifact.name}@{artifact.version} holds no file or directory to download. "
            "Read its value with `to_python()`."
        )
    return value.scalar.blob


def _is_dir(blob: literals_pb2.Blob) -> bool:
    return blob.metadata.type.dimensionality == types_pb2.BlobType.MULTIPART


def default_destination(artifact: Artifact) -> pathlib.Path:
    """Where `download` writes when no destination is given: the blob's own name under the cwd."""
    uri = _blob(artifact).uri.rstrip("/")
    return pathlib.Path.cwd() / (posixpath.basename(uri) or artifact.name)


def _safe_join(root: pathlib.Path, relative_path: str) -> pathlib.Path:
    """Join a server-reported object path under root, refusing anything that would escape it."""
    parts = pathlib.PurePosixPath(relative_path).parts
    if not parts or pathlib.PurePosixPath(relative_path).is_absolute() or ".." in parts:
        raise RuntimeError(f"Refusing to write object with unsafe relative path {relative_path!r}.")
    return root.joinpath(*parts)


def _supports_artifact_links() -> bool:
    fields = dataproxy_service_pb2.CreateDownloadLinkRequest.DESCRIPTOR.fields_by_name
    response_fields = dataproxy_service_pb2.CreateDownloadLinkResponse.DESCRIPTOR.fields_by_name
    return (
        "artifact_id" in fields
        and "relative_paths" in response_fields
        and hasattr(dataproxy_service_pb2, "ARTIFACT_TYPE_VALUE")
    )


async def _signed_links(artifact: Artifact) -> tuple[list[str], list[str]]:
    """Ask the data proxy to sign the artifact's data. Returns (urls, relative paths)."""
    if not _supports_artifact_links():
        raise ArtifactDownloadNotSupported(
            "the installed flyteidl2 can't request download links for artifacts; upgrade flyteidl2"
        )
    ensure_client()
    resp = await get_client().dataproxy_service.create_download_link(value_link_request(artifact))
    return list(resp.pre_signed_urls.signed_url), list(resp.relative_paths)  # type: ignore[attr-defined]


def value_link_request(artifact: Artifact) -> dataproxy_service_pb2.CreateDownloadLinkRequest:
    """The CreateDownloadLink request that signs the data behind `artifact`'s value."""
    request = dataproxy_service_pb2.CreateDownloadLinkRequest(
        artifact_type=dataproxy_service_pb2.ARTIFACT_TYPE_VALUE,  # type: ignore[attr-defined]
    )
    request.artifact_id.CopyFrom(artifact.pb2.artifact_id)  # type: ignore[attr-defined]
    return request


async def _fetch(client: httpx.AsyncClient, url: str, target: pathlib.Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    async with client.stream("GET", url) as response:
        response.raise_for_status()
        with target.open("wb") as f:
            async for chunk in response.aiter_bytes():
                f.write(chunk)


async def _download_via_data_proxy(artifact: Artifact, is_dir: bool, dest: pathlib.Path) -> None:
    urls, relative_paths = await _signed_links(artifact)
    if not is_dir:
        if len(urls) != 1:
            raise RuntimeError(f"The data proxy returned {len(urls)} download links for a file, expected 1.")
        targets = [dest]
    else:
        if len(relative_paths) != len(urls):
            raise RuntimeError(
                f"The data proxy returned {len(urls)} download links but {len(relative_paths)} object paths "
                "for a directory."
            )
        targets = [_safe_join(dest, rel) for rel in relative_paths]
        dest.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
        for url, target in zip(urls, targets):
            await _fetch(client, url, target)


async def _download_via_storage(uri: str, is_dir: bool, dest: pathlib.Path) -> None:
    import flyte.storage as storage

    if is_dir:
        dest.mkdir(parents=True, exist_ok=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
    await storage.get(uri, str(dest), recursive=is_dir)


async def download_artifact(artifact: Artifact, dest: pathlib.Path) -> pathlib.Path:
    """
    Download the file or directory behind `artifact` to `dest` and return the local path.

    A file is written to `dest`, or inside it when `dest` is an existing directory. A directory's
    objects are written under `dest`, keeping their paths.
    """
    blob = _blob(artifact)
    is_dir = _is_dir(blob)
    if not is_dir and dest.is_dir():
        dest = dest / (posixpath.basename(blob.uri.rstrip("/")) or artifact.name)

    try:
        await _download_via_data_proxy(artifact, is_dir, dest)
        return dest
    except Exception as proxy_error:
        logger.debug(
            f"Could not download artifact {artifact.name}@{artifact.version} through the data proxy "
            f"({proxy_error}), reading {blob.uri} directly."
        )
        try:
            await _download_via_storage(blob.uri, is_dir, dest)
        except Exception as storage_error:
            raise RuntimeError(
                f"Could not download artifact {artifact.name}@{artifact.version}. The data proxy could not "
                f"sign it ({proxy_error}), and reading {blob.uri} directly failed ({storage_error}). Reading "
                "the object store directly needs storage credentials, which a machine outside the cluster "
                "usually doesn't have; signed downloads need a backend that supports them."
            ) from storage_error
    return dest
