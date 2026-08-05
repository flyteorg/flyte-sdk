"""Signed-PUT uploads for tracked-run metadata artifacts (inputs.pb / outputs.pb / report.html).

Tracked-run artifacts are uploaded via ``DataProxyService.CreateUploadLocation`` routed
through SelectCluster's ``OPERATION_TRACKED_RUN_DATA`` (the storage path is derived from a
deterministic ``tracked-runs/<run>/<action>[/<attempt>]`` filename root), followed by an
HTTP PUT to the returned signed URL. Unlike ``flyte.remote._data._upload_single_file``
this uploads in-memory bytes — the local runtime holds inputs/outputs protos in memory
and reports are small HTML files.
"""

from __future__ import annotations

import hashlib
import typing
from base64 import b64encode
from datetime import timedelta

if typing.TYPE_CHECKING:
    from flyteidl2.common import identifier_pb2

    from flyte.remote._client._protocols import DataProxyService

_UPLOAD_EXPIRES_IN = timedelta(seconds=60)

# Artifact kind -> stored filename. The kind doubles as the caller-facing vocabulary
# (the old dataproxy ArtifactType INPUTS/OUTPUTS enum values are gone).
_KIND_FILENAMES = {
    "inputs": "inputs.pb",
    "outputs": "outputs.pb",
    "report": "report.html",
}


async def upload_tracked_run_artifact(
    dataproxy: DataProxyService,
    *,
    kind: str,
    run_id: identifier_pb2.RunIdentifier,
    action_name: str,
    attempt: int | None,
    data: bytes,
    verify: bool = True,
    max_retries: int = 3,
    content_type: str | None = None,
) -> tuple[str, str]:
    """Upload one tracked-run metadata artifact, returning ``(native_url, cluster)``.

    :param dataproxy: The cluster-aware dataproxy client to request the signed URL from
        (routed via SelectCluster's ``OPERATION_TRACKED_RUN_DATA``).
    :param kind: ``"inputs"``, ``"outputs"`` or ``"report"``. Inputs target an action
        (``attempt`` must be None); outputs / reports target an action attempt.
    :param run_id: The tracked run the artifact belongs to (org/project/domain/name).
    :param action_name: Target action name (e.g. ``a0``).
    :param attempt: Target attempt for outputs / reports; None for inputs.
    :param data: The artifact content (serialized proto bytes or report HTML bytes).
    :param verify: Whether to verify TLS certificates on the signed PUT.
    :param max_retries: Maximum retry attempts for the signed PUT.
    :param content_type: Optional Content-Type stored as object metadata (e.g. ``text/html``
        for reports so browsers render instead of download them). The dataproxy's presigned
        URLs do not sign the content type, so sending the header is safe.
    :return: ``(native_url, cluster)`` — the native storage URL (e.g. ``s3://...``) of the
        uploaded artifact and the routing cluster's name ("" when served by the control
        plane), for stamping on reported attempt events.
    """
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError
    from flyteidl2.dataproxy import dataproxy_service_pb2

    from flyte.errors import RuntimeSystemError

    filename = _KIND_FILENAMES.get(kind)
    if filename is None:
        raise ValueError(f"Unknown tracked-run artifact kind {kind!r}; expected one of {sorted(_KIND_FILENAMES)}")
    if (kind == "inputs") != (attempt is None):
        raise ValueError("attempt must be None for 'inputs' and set for 'outputs' / 'report'")

    filename_root = f"tracked-runs/{run_id.name}/{action_name}"
    if attempt is not None:
        filename_root = f"{filename_root}/{attempt}"

    md5_bytes = hashlib.md5(data).digest()
    req = dataproxy_service_pb2.CreateUploadLocationRequest(
        org=run_id.org,
        project=run_id.project,
        domain=run_id.domain,
        filename_root=filename_root,
        filename=filename,
        content_md5=md5_bytes,
        content_length=len(data),
        add_content_md5_metadata=True,
    )
    req.expires_in.FromTimedelta(_UPLOAD_EXPIRES_IN)
    try:
        resp, cluster = await dataproxy.create_tracked_run_upload_location(req)
    except ConnectError as e:
        if e.code == Code.UNAVAILABLE:
            raise RuntimeSystemError(
                "SystemUnavailableError",
                f"CreateUploadLocation({kind}) failed: service unavailable. {e.message}",
            ) from e
        raise RuntimeSystemError(e.code.value, f"CreateUploadLocation({kind}) failed: {e.message}") from e
    except Exception as e:
        raise RuntimeSystemError(type(e).__name__, f"CreateUploadLocation({kind}) failed: {e}") from e

    from flyte.remote._data import get_extra_headers_for_protocol

    headers = get_extra_headers_for_protocol(resp.native_url)
    headers.update(resp.headers)
    headers.update(
        {
            "Content-Length": str(len(data)),
            "Content-MD5": b64encode(md5_bytes).decode("utf-8"),
        }
    )
    if content_type:
        headers["Content-Type"] = content_type
    await _put_bytes_with_retry(
        data,
        signed_url=resp.signed_url,
        extra_headers=headers,
        verify=verify,
        max_retries=max_retries,
    )
    return resp.native_url, cluster


async def _put_bytes_with_retry(
    data: bytes,
    *,
    signed_url: str,
    extra_headers: dict,
    verify: bool = True,
    max_retries: int = 3,
    min_backoff_sec: float = 0.5,
    max_backoff_sec: float = 10.0,
    retry_after_cap_sec: float = 60.0,
) -> None:
    """PUT in-memory bytes to a signed URL with exponential backoff retry.

    Thin wrapper over ``flyte.remote._data._put_signed_url_with_retry`` (shared with
    file uploads: retryable status codes, Retry-After handling, signed-URL redaction).
    Raises ``RuntimeSystemError`` when the upload ultimately fails.
    """
    from flyte.remote._data import _put_signed_url_with_retry

    await _put_signed_url_with_retry(
        data,
        signed_url,
        extra_headers,
        verify,
        max_retries=max_retries,
        min_backoff_sec=min_backoff_sec,
        max_backoff_sec=max_backoff_sec,
        retry_after_cap_sec=retry_after_cap_sec,
    )
