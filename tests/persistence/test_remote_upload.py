"""Tests for the tracked-run metadata signed-PUT upload helper."""

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

from flyte._persistence import _remote_upload
from flyte._persistence._remote_upload import (
    _create_upload_location_with_retry,
    _put_bytes_with_retry,
    upload_tracked_run_artifact,
)
from flyte.errors import RuntimeSystemError

_RUN_ID = identifier_pb2.RunIdentifier(org="o", project="p", domain="d", name="local-x")


def _data_module():
    """Import `flyte.remote._data` lazily, the way the upload helper itself does."""
    from flyte.remote import _data

    return _data


def _make_dataproxy(headers: dict | None = None, cluster: str = ""):
    dataproxy = MagicMock()
    dataproxy.create_tracked_run_upload_location = AsyncMock(
        return_value=(
            dataproxy_service_pb2.CreateUploadLocationResponse(
                signed_url="https://signed.example/put?sig=secret",
                native_url="s3://bucket/tracked-runs/local-x/a0/inputs.pb",
                headers=headers or {},
            ),
            cluster,
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
async def test_upload_tracked_run_artifact_success():
    data = b"serialized-proto-bytes"
    dataproxy = _make_dataproxy(headers={"x-extra": "1"}, cluster="cluster-a")
    client, ctx = _mock_http_client([httpx.Response(200)])

    with patch("httpx.AsyncClient", return_value=ctx):
        native_url, cluster = await upload_tracked_run_artifact(
            dataproxy,
            kind="inputs",
            run_id=_RUN_ID,
            action_name="a0",
            attempt=None,
            data=data,
        )

    assert native_url == "s3://bucket/tracked-runs/local-x/a0/inputs.pb"
    assert cluster == "cluster-a"

    req = dataproxy.create_tracked_run_upload_location.await_args[0][0]
    assert req.org == "o"
    assert req.project == "p"
    assert req.domain == "d"
    assert req.filename_root == "tracked-runs/local-x/a0"
    assert req.filename == "inputs.pb"
    assert req.content_md5 == hashlib.md5(data).digest()
    assert req.content_length == len(data)
    assert req.add_content_md5_metadata is True
    assert req.expires_in.ToTimedelta() == _data_module()._UPLOAD_EXPIRES_IN

    client.put.assert_awaited_once()
    put_call = client.put.await_args
    assert put_call[0][0] == "https://signed.example/put?sig=secret"
    sent_headers = put_call[1]["headers"]
    assert sent_headers["x-extra"] == "1"  # response headers are honored
    assert sent_headers["Content-Length"] == str(len(data))
    assert sent_headers["Content-MD5"] == b64encode(hashlib.md5(data).digest()).decode("utf-8")
    assert put_call[1]["content"] == data


@pytest.mark.asyncio
async def test_upload_tracked_run_artifact_outputs_target_attempt():
    dataproxy = _make_dataproxy()
    _, ctx = _mock_http_client([httpx.Response(204)])

    with patch("httpx.AsyncClient", return_value=ctx):
        _, cluster = await upload_tracked_run_artifact(
            dataproxy,
            kind="outputs",
            run_id=_RUN_ID,
            action_name="a0",
            attempt=2,
            data=b"outputs",
        )

    # Control-plane-served uploads report no routing cluster.
    assert cluster == ""
    req = dataproxy.create_tracked_run_upload_location.await_args[0][0]
    assert req.filename_root == "tracked-runs/local-x/a0/2"
    assert req.filename == "outputs.pb"


@pytest.mark.asyncio
async def test_upload_tracked_run_artifact_report_filename():
    dataproxy = _make_dataproxy()
    _, ctx = _mock_http_client([httpx.Response(200)])

    with patch("httpx.AsyncClient", return_value=ctx):
        await upload_tracked_run_artifact(
            dataproxy,
            kind="report",
            run_id=_RUN_ID,
            action_name="a1",
            attempt=1,
            data=b"<html/>",
            content_type="text/html",
        )

    req = dataproxy.create_tracked_run_upload_location.await_args[0][0]
    assert req.filename_root == "tracked-runs/local-x/a1/1"
    assert req.filename == "report.html"


@pytest.mark.asyncio
async def test_upload_tracked_run_artifact_validates_kind_and_attempt():
    dataproxy = _make_dataproxy()
    # Unknown kind.
    with pytest.raises(ValueError, match="kind"):
        await upload_tracked_run_artifact(
            dataproxy, kind="code", run_id=_RUN_ID, action_name="a0", attempt=None, data=b"x"
        )
    # Inputs must not carry an attempt.
    with pytest.raises(ValueError, match="attempt"):
        await upload_tracked_run_artifact(
            dataproxy, kind="inputs", run_id=_RUN_ID, action_name="a0", attempt=1, data=b"x"
        )
    # Outputs / reports must carry an attempt.
    with pytest.raises(ValueError, match="attempt"):
        await upload_tracked_run_artifact(
            dataproxy, kind="outputs", run_id=_RUN_ID, action_name="a0", attempt=None, data=b"x"
        )
    dataproxy.create_tracked_run_upload_location.assert_not_awaited()


@pytest.mark.asyncio
async def test_upload_tracked_run_artifact_maps_connect_errors():
    dataproxy = MagicMock()
    dataproxy.create_tracked_run_upload_location = AsyncMock(
        side_effect=ConnectError(Code.PERMISSION_DENIED, "not yours")
    )

    with pytest.raises(RuntimeSystemError, match="not yours") as exc:
        await upload_tracked_run_artifact(
            dataproxy,
            kind="inputs",
            run_id=_RUN_ID,
            action_name="a0",
            attempt=None,
            data=b"x",
        )

    # Non-transient codes are terminal on the first attempt.
    assert exc.value.code == "permission_denied"
    dataproxy.create_tracked_run_upload_location.assert_awaited_once()


_UPLOAD_OK = (
    dataproxy_service_pb2.CreateUploadLocationResponse(
        signed_url="https://signed.example/put",
        native_url="s3://bucket/tracked-runs/local-x/a0/inputs.pb",
    ),
    "cluster-a",
)

_UPLOAD_REQ = dataproxy_service_pb2.CreateUploadLocationRequest(
    org="o", project="p", domain="d", filename_root="tracked-runs/local-x/a0", filename="inputs.pb"
)


@pytest.mark.asyncio
async def test_upload_tracked_run_artifact_retries_gateway_timeout(monkeypatch):
    """SDK-60: a transient 504 (DEADLINE_EXCEEDED) on CreateUploadLocation must be retried
    before the whole tracked-run upload is failed, like the signed PUT already does."""
    monkeypatch.setattr(_remote_upload, "_RPC_MIN_BACKOFF_SEC", 0.001)
    dataproxy = MagicMock()
    dataproxy.create_tracked_run_upload_location = AsyncMock(
        side_effect=[ConnectError(Code.DEADLINE_EXCEEDED, "Gateway Timeout"), _UPLOAD_OK]
    )
    client, ctx = _mock_http_client([httpx.Response(200)])

    with patch("httpx.AsyncClient", return_value=ctx):
        native_url, cluster = await upload_tracked_run_artifact(
            dataproxy, kind="inputs", run_id=_RUN_ID, action_name="a0", attempt=None, data=b"x"
        )

    assert native_url == "s3://bucket/tracked-runs/local-x/a0/inputs.pb"
    assert cluster == "cluster-a"
    assert dataproxy.create_tracked_run_upload_location.await_count == 2
    client.put.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [Code.UNAVAILABLE, Code.INTERNAL, Code.DEADLINE_EXCEEDED, Code.UNKNOWN])
async def test_create_upload_location_retries_transient_codes(code):
    dataproxy = MagicMock()
    dataproxy.create_tracked_run_upload_location = AsyncMock(
        side_effect=[ConnectError(code, "flaky"), ConnectError(code, "flaky"), _UPLOAD_OK]
    )

    resp, cluster = await _create_upload_location_with_retry(
        dataproxy, _UPLOAD_REQ, kind="inputs", max_retries=3, min_backoff_sec=0.001
    )

    assert resp.signed_url == "https://signed.example/put"
    assert cluster == "cluster-a"
    assert dataproxy.create_tracked_run_upload_location.await_count == 3


@pytest.mark.asyncio
async def test_create_upload_location_gives_up_after_max_retries():
    dataproxy = MagicMock()
    dataproxy.create_tracked_run_upload_location = AsyncMock(
        side_effect=ConnectError(Code.DEADLINE_EXCEEDED, "Gateway Timeout")
    )

    with pytest.raises(RuntimeSystemError, match="after 2 retries: Gateway Timeout") as exc:
        await _create_upload_location_with_retry(
            dataproxy, _UPLOAD_REQ, kind="outputs", max_retries=2, min_backoff_sec=0.001
        )

    assert exc.value.code == "deadline_exceeded"
    assert dataproxy.create_tracked_run_upload_location.await_count == 3


@pytest.mark.asyncio
async def test_create_upload_location_exhausted_unavailable_keeps_its_code():
    dataproxy = MagicMock()
    dataproxy.create_tracked_run_upload_location = AsyncMock(side_effect=ConnectError(Code.UNAVAILABLE, "down"))

    with pytest.raises(RuntimeSystemError, match="service unavailable") as exc:
        await _create_upload_location_with_retry(
            dataproxy, _UPLOAD_REQ, kind="inputs", max_retries=1, min_backoff_sec=0.001
        )

    assert exc.value.code == "SystemUnavailableError"
    assert dataproxy.create_tracked_run_upload_location.await_count == 2


@pytest.mark.asyncio
async def test_create_upload_location_retries_wrapped_select_cluster_error():
    """The cluster-aware dataproxy wraps SelectCluster failures in a RuntimeError chained to
    the ConnectError; the transient code is still found through the cause chain."""
    wrapped = RuntimeError("SelectCluster failed for OPERATION_TRACKED_RUN_DATA: unavailable")
    wrapped.__cause__ = ConnectError(Code.UNAVAILABLE, "unavailable")
    dataproxy = MagicMock()
    dataproxy.create_tracked_run_upload_location = AsyncMock(side_effect=[wrapped, _UPLOAD_OK])

    await _create_upload_location_with_retry(dataproxy, _UPLOAD_REQ, kind="inputs", min_backoff_sec=0.001)

    assert dataproxy.create_tracked_run_upload_location.await_count == 2


@pytest.mark.asyncio
async def test_create_upload_location_does_not_retry_plain_exceptions():
    dataproxy = MagicMock()
    dataproxy.create_tracked_run_upload_location = AsyncMock(side_effect=RuntimeError("no storage"))

    with pytest.raises(RuntimeSystemError, match="no storage") as exc:
        await _create_upload_location_with_retry(dataproxy, _UPLOAD_REQ, kind="inputs", min_backoff_sec=0.001)

    assert exc.value.code == "RuntimeError"
    dataproxy.create_tracked_run_upload_location.assert_awaited_once()


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


@pytest.mark.asyncio
async def test_tracked_run_signed_url_outlives_the_put_it_authorizes():
    """The signed URL must stay valid at least as long as the PUT it authorizes.

    Tracked-run metadata goes out through the very same `_put_signed_url_with_retry` as
    file uploads, which allows a full `FLYTE_UPLOAD_TIMEOUT` per attempt and then retries.
    This call site used to mint its URL with a hardcoded 60s lifetime, so the signature
    could lapse ten times over before a single attempt even timed out -- the failure the
    file-upload path was fixed for in #1363 (FLYTE-SDK-5F / 4B / 6C / 7G / 7H / 7K).

    Pinning the derived constant instead of a literal is the point: it is what stops the
    two paths that share one PUT helper from drifting apart again, and it carries the
    FLYTE_UPLOAD_EXPIRES_IN override to this path for free.
    """
    data_mod = _data_module()
    dataproxy = _make_dataproxy()
    _, ctx = _mock_http_client([httpx.Response(200)])

    with patch("httpx.AsyncClient", return_value=ctx):
        await upload_tracked_run_artifact(
            dataproxy,
            kind="inputs",
            run_id=_RUN_ID,
            action_name="a0",
            attempt=None,
            data=b"serialized-proto-bytes",
        )

    req = dataproxy.create_tracked_run_upload_location.await_args[0][0]
    signed_for = req.expires_in.ToTimedelta().total_seconds()

    assert signed_for == data_mod._UPLOAD_EXPIRES_IN.total_seconds()
    assert signed_for >= data_mod._UPLOAD_TIMEOUT_SECONDS, (
        f"signed URL lives {signed_for}s but one PUT attempt may run for {data_mod._UPLOAD_TIMEOUT_SECONDS}s"
    )
