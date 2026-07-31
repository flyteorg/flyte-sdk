"""Signed-PUT uploads for local-run metadata artifacts (inputs.pb / outputs.pb / report.html).

Local runs are tracked exclusively by the control plane: artifacts are uploaded via
``DataProxyService.UploadMetadata`` (which derives the storage path from the target
action / action attempt) followed by an HTTP PUT to the returned signed URL. Unlike
``flyte.remote._data._upload_single_file`` this uploads in-memory bytes — the local
runtime holds inputs/outputs protos in memory and reports are small HTML files.
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


async def upload_metadata_artifact(
    dataproxy: DataProxyService,
    *,
    artifact_type: int,
    data: bytes,
    action_id: identifier_pb2.ActionIdentifier | None = None,
    action_attempt_id: identifier_pb2.ActionAttemptIdentifier | None = None,
    verify: bool = True,
    max_retries: int = 3,
) -> str:
    """Upload one local-run metadata artifact and return its native (storage) URL.

    :param dataproxy: The (plain, control-plane) dataproxy client to request the signed URL from.
    :param artifact_type: ``dataproxy_service_pb2.ArtifactType`` value — INPUTS targets an
        action, OUTPUTS / REPORT target an action attempt.
    :param data: The artifact content (serialized proto bytes or report HTML bytes).
    :param action_id: Target action (required for INPUTS).
    :param action_attempt_id: Target action attempt (required for OUTPUTS / REPORT).
    :param verify: Whether to verify TLS certificates on the signed PUT.
    :param max_retries: Maximum retry attempts for the signed PUT.
    :return: The native storage URL (e.g. ``s3://...``) of the uploaded artifact.
    """
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError
    from flyteidl2.dataproxy import dataproxy_service_pb2

    from flyte.errors import RuntimeSystemError

    if (action_id is None) == (action_attempt_id is None):
        raise ValueError("Exactly one of action_id or action_attempt_id must be provided")

    md5_bytes = hashlib.md5(data).digest()
    req = dataproxy_service_pb2.UploadMetadataRequest(
        artifact_type=artifact_type,  # type: ignore[arg-type]
        content_md5=md5_bytes,
        content_length=len(data),
        add_content_md5_metadata=True,
    )
    req.expires_in.FromTimedelta(_UPLOAD_EXPIRES_IN)
    if action_id is not None:
        req.action_id.CopyFrom(action_id)
    else:
        req.action_attempt_id.CopyFrom(action_attempt_id)

    artifact_name = dataproxy_service_pb2.ArtifactType.Name(artifact_type)
    try:
        resp = await dataproxy.upload_metadata(req)
    except ConnectError as e:
        if e.code == Code.UNAVAILABLE:
            raise RuntimeSystemError(
                "SystemUnavailableError",
                f"UploadMetadata({artifact_name}) failed: service unavailable. {e.message}",
            ) from e
        raise RuntimeSystemError(e.code.value, f"UploadMetadata({artifact_name}) failed: {e.message}") from e
    except Exception as e:
        raise RuntimeSystemError(type(e).__name__, f"UploadMetadata({artifact_name}) failed: {e}") from e

    from flyte.remote._data import get_extra_headers_for_protocol

    headers = get_extra_headers_for_protocol(resp.native_url)
    headers.update(resp.headers)
    headers.update(
        {
            "Content-Length": str(len(data)),
            "Content-MD5": b64encode(md5_bytes).decode("utf-8"),
        }
    )
    await _put_bytes_with_retry(
        data,
        signed_url=resp.signed_url,
        extra_headers=headers,
        verify=verify,
        max_retries=max_retries,
    )
    return resp.native_url


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
