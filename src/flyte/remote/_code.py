"""
Fetching the code bundle a remote run executed.

When a task is launched from a local directory, the SDK packages the source into a
`flyte.models.CodeBundle` (a tarball, or a cloudpickle for notebook/REPL sessions), uploads it,
and records its location in the task spec. This module is the read path back: given an action,
find that bundle and pull it down — the same source the console's "Code" tab shows.

Two routes to the bytes, tried in this order:

1. **The data proxy.** `CreateDownloadLink(ARTIFACT_TYPE_CODE_BUNDLE)` returns a pre-signed URL.
   This is what the console uses, and it needs no object-store credentials on the client — only
   the Flyte API key/session already in use. Backends that don't serve this artifact type
   (OSS Flyte signs reports only) and pickled bundles (deliberately refused, they aren't source)
   fail here and fall through.
2. **Direct object-store read** of the URI recorded in the task spec, via `flyte.storage`. Works
   wherever the caller has credentials for the bucket — inside a cluster, or locally when the
   blob store is configured — and is the only route for a pkl bundle.

If both fail the data-proxy error is raised, with the storage error chained onto it, so the
message explains the route the caller most likely expected to work.
"""

from __future__ import annotations

import pathlib

import httpx
from flyteidl2.common import identifier_pb2
from flyteidl2.dataproxy import dataproxy_service_pb2

from flyte._initialize import ensure_client, get_client
from flyte._logging import logger
from flyte.models import CodeBundle

# Code bundles are source archives — small — but "small" is a matter of copy_style, and the
# default httpx timeout is too tight for a bundle that ships data files alongside the code.
_DOWNLOAD_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=10.0)


def bundle_uri(bundle: CodeBundle) -> str:
    """The remote location of the bundle's archive, tgz or pkl."""
    uri = bundle.tgz or bundle.pkl
    if not uri:
        raise ValueError("Code bundle should be either tgz or pkl, found neither.")
    return uri


async def _download_via_data_proxy(
    action_id: identifier_pb2.ActionIdentifier,
    attempt: int,
    target: pathlib.Path,
) -> None:
    """Sign a download link for the action's code bundle and stream it into target."""
    ensure_client()
    resp = await get_client().dataproxy_service.create_download_link(
        dataproxy_service_pb2.CreateDownloadLinkRequest(
            artifact_type=dataproxy_service_pb2.ARTIFACT_TYPE_CODE_BUNDLE,
            action_attempt_id=identifier_pb2.ActionAttemptIdentifier(
                action_id=action_id,
                attempt=attempt,
            ),
        )
    )
    signed_urls = list(resp.pre_signed_urls.signed_url)
    if not signed_urls:
        raise RuntimeError("The data proxy returned no download link for the code bundle.")

    async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
        async with client.stream("GET", signed_urls[0]) as response:
            response.raise_for_status()
            with target.open("wb") as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)


async def _download_via_storage(bundle: CodeBundle, target: pathlib.Path) -> None:
    """Read the bundle straight out of the blob store recorded in the task spec."""
    import flyte.storage as storage

    await storage.get(bundle_uri(bundle), str(target))


async def download_code_bundle(
    bundle: CodeBundle,
    dest: pathlib.Path,
    *,
    action_id: identifier_pb2.ActionIdentifier,
    attempt: int,
    extract: bool = True,
) -> pathlib.Path:
    """
    Download `bundle` into `dest`, optionally extracting a tarball.

    Args:
        bundle: The bundle to fetch, as recorded in the action's task spec.
        dest: Directory to download into. Created if it does not exist.
        action_id: The action the bundle belongs to, used to sign the download link.
        attempt: The attempt to sign the download link for.
        extract: Unpack a tgz bundle into `dest` after downloading. Pkl bundles are never
            unpacked — there is nothing to unpack.

    Returns:
        The directory the source was extracted into, or the path of the downloaded archive when
        it was not extracted.
    """
    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if not dest.is_dir():
        raise ValueError(f"Destination should be a directory, found {dest}")

    uri = bundle_uri(bundle)
    archive = dest / pathlib.PurePosixPath(uri).name

    try:
        await _download_via_data_proxy(action_id, attempt, archive)
    except Exception as proxy_error:
        logger.debug(f"Could not fetch code bundle via the data proxy ({proxy_error}), reading {uri} directly.")
        try:
            await _download_via_storage(bundle, archive)
        except Exception as storage_error:
            archive.unlink(missing_ok=True)
            raise proxy_error from storage_error

    if extract and bundle.tgz:
        # The same extraction the worker performs when it inflates a bundle in-cluster.
        from flyte._code_bundle.bundle import _extract_tar

        await _extract_tar(archive, dest)
        return dest

    return archive
