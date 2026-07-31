"""Tests for the local-run metadata signed-PUT upload helper."""

from __future__ import annotations

import hashlib
from base64 import b64encode
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from flyteidl2.common import identifier_pb2
from flyteidl2.dataproxy import dataproxy_service_pb2

from flyte._persistence._remote_upload import _put_bytes_with_retry, upload_metadata_artifact
from flyte.errors import RuntimeSystemError

_RUN_ID = identifier_pb2.RunIdentifier(org="o", project="p", domain="d", name="local-x")
_ACTION_ID = identifier_pb2.ActionIdentifier(run=_RUN_ID, name="a0")
_ATTEMPT_ID = identifier_pb2.ActionAttemptIdentifier(action_id=_ACTION_ID, attempt=1)


def _make_dataproxy(headers: dict | None = None):
    dataproxy = MagicMock()
    dataproxy.upload_metadata = AsyncMock(
        return_value=dataproxy_service_pb2.CreateUploadLocationResponse(
            signed_url="https://signed.example/put?sig=secret",
            native_url="s3://bucket/org/p/d/local-x/a0/inputs.pb",
            headers=headers or {},
        )
    )
    return dataproxy


def _mock_http_client(put_results):
    """Build a patched httpx.AsyncClient whose put() yields the given results in order."""
    client = AsyncMock()
    client.put.side_effect = put_results
    ctx = AsyncMock()
    ctx.__aenter__.return_value = client
    ctx.__aexit__.return_value = False
    return client, ctx


@pytest.mark.asyncio
async def test_upload_metadata_artifact_success():
    data = b"serialized-proto-bytes"
    dataproxy = _make_dataproxy(headers={"x-extra": "1"})
    client, ctx = _mock_http_client([httpx.Response(200)])

    with patch("httpx.AsyncClient", return_value=ctx):
        native_url = await upload_metadata_artifact(
            dataproxy,
            artifact_type=int(dataproxy_service_pb2.ARTIFACT_TYPE_INPUTS),
            data=data,
            action_id=_ACTION_ID,
        )

    assert native_url == "s3://bucket/org/p/d/local-x/a0/inputs.pb"

    req = dataproxy.upload_metadata.await_args[0][0]
    assert req.artifact_type == dataproxy_service_pb2.ARTIFACT_TYPE_INPUTS
    assert req.WhichOneof("target") == "action_id"
    assert req.action_id == _ACTION_ID
    assert req.content_md5 == hashlib.md5(data).digest()
    assert req.content_length == len(data)
    assert req.add_content_md5_metadata is True

    client.put.assert_awaited_once()
    put_call = client.put.await_args
    assert put_call[0][0] == "https://signed.example/put?sig=secret"
    sent_headers = put_call[1]["headers"]
    assert sent_headers["x-extra"] == "1"  # response headers are honored
    assert sent_headers["Content-Length"] == str(len(data))
    assert sent_headers["Content-MD5"] == b64encode(hashlib.md5(data).digest()).decode("utf-8")
    assert put_call[1]["content"] == data


@pytest.mark.asyncio
async def test_upload_metadata_artifact_targets_attempt():
    dataproxy = _make_dataproxy()
    _, ctx = _mock_http_client([httpx.Response(204)])

    with patch("httpx.AsyncClient", return_value=ctx):
        await upload_metadata_artifact(
            dataproxy,
            artifact_type=int(dataproxy_service_pb2.ARTIFACT_TYPE_OUTPUTS),
            data=b"outputs",
            action_attempt_id=_ATTEMPT_ID,
        )

    req = dataproxy.upload_metadata.await_args[0][0]
    assert req.WhichOneof("target") == "action_attempt_id"
    assert req.action_attempt_id == _ATTEMPT_ID


@pytest.mark.asyncio
async def test_upload_metadata_artifact_requires_exactly_one_target():
    dataproxy = _make_dataproxy()
    with pytest.raises(ValueError):
        await upload_metadata_artifact(
            dataproxy, artifact_type=int(dataproxy_service_pb2.ARTIFACT_TYPE_INPUTS), data=b"x"
        )
    with pytest.raises(ValueError):
        await upload_metadata_artifact(
            dataproxy,
            artifact_type=int(dataproxy_service_pb2.ARTIFACT_TYPE_INPUTS),
            data=b"x",
            action_id=_ACTION_ID,
            action_attempt_id=_ATTEMPT_ID,
        )


@pytest.mark.asyncio
async def test_upload_metadata_artifact_maps_connect_errors():
    dataproxy = MagicMock()
    dataproxy.upload_metadata = AsyncMock(side_effect=ConnectError(Code.PERMISSION_DENIED, "not yours"))

    with pytest.raises(RuntimeSystemError, match="not yours"):
        await upload_metadata_artifact(
            dataproxy,
            artifact_type=int(dataproxy_service_pb2.ARTIFACT_TYPE_INPUTS),
            data=b"x",
            action_id=_ACTION_ID,
        )


@pytest.mark.asyncio
async def test_put_bytes_retries_then_succeeds():
    client, ctx = _mock_http_client([httpx.Response(503), httpx.Response(200)])

    with patch("httpx.AsyncClient", return_value=ctx):
        await _put_bytes_with_retry(
            b"data",
            signed_url="https://signed.example/put",
            extra_headers={},
            max_retries=2,
            min_backoff_sec=0.01,
        )

    assert client.put.await_count == 2


@pytest.mark.asyncio
async def test_put_bytes_gives_up_after_max_retries():
    client, ctx = _mock_http_client([httpx.Response(500)] * 3)

    with patch("httpx.AsyncClient", return_value=ctx):
        with pytest.raises(RuntimeSystemError, match="after 2 retries"):
            await _put_bytes_with_retry(
                b"data",
                signed_url="https://signed.example/put",
                extra_headers={},
                max_retries=2,
                min_backoff_sec=0.01,
            )

    assert client.put.await_count == 3


@pytest.mark.asyncio
async def test_put_bytes_does_not_retry_client_errors_and_redacts_url():
    client, ctx = _mock_http_client([httpx.Response(403)])

    with patch("httpx.AsyncClient", return_value=ctx):
        with pytest.raises(RuntimeSystemError) as exc:
            await _put_bytes_with_retry(
                b"data",
                signed_url="https://signed.example/put?X-Amz-Signature=secret",
                extra_headers={},
                max_retries=3,
                min_backoff_sec=0.01,
            )

    assert client.put.await_count == 1
    assert "X-Amz-Signature=secret" not in str(exc.value)
    assert "<redacted>" in str(exc.value)


@pytest.mark.asyncio
async def test_put_bytes_retries_on_network_error():
    client, ctx = _mock_http_client([httpx.ConnectError("refused"), httpx.Response(201)])

    with patch("httpx.AsyncClient", return_value=ctx):
        await _put_bytes_with_retry(
            b"data",
            signed_url="https://signed.example/put",
            extra_headers={},
            max_retries=1,
            min_backoff_sec=0.01,
        )

    assert client.put.await_count == 2
