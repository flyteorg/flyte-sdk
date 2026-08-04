"""Tests for the traced-run metadata signed-PUT upload helper."""

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

from flyte._persistence._remote_upload import _put_bytes_with_retry, upload_traced_run_artifact
from flyte.errors import RuntimeSystemError

_RUN_ID = identifier_pb2.RunIdentifier(org="o", project="p", domain="d", name="local-x")


def _make_dataproxy(headers: dict | None = None, cluster: str = ""):
    dataproxy = MagicMock()
    dataproxy.create_traced_run_upload_location = AsyncMock(
        return_value=(
            dataproxy_service_pb2.CreateUploadLocationResponse(
                signed_url="https://signed.example/put?sig=secret",
                native_url="s3://bucket/traced-runs/local-x/a0/inputs.pb",
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
async def test_upload_traced_run_artifact_success():
    data = b"serialized-proto-bytes"
    dataproxy = _make_dataproxy(headers={"x-extra": "1"}, cluster="cluster-a")
    client, ctx = _mock_http_client([httpx.Response(200)])

    with patch("httpx.AsyncClient", return_value=ctx):
        native_url, cluster = await upload_traced_run_artifact(
            dataproxy,
            kind="inputs",
            run_id=_RUN_ID,
            action_name="a0",
            attempt=None,
            data=data,
        )

    assert native_url == "s3://bucket/traced-runs/local-x/a0/inputs.pb"
    assert cluster == "cluster-a"

    req = dataproxy.create_traced_run_upload_location.await_args[0][0]
    assert req.org == "o"
    assert req.project == "p"
    assert req.domain == "d"
    assert req.filename_root == "traced-runs/local-x/a0"
    assert req.filename == "inputs.pb"
    assert req.content_md5 == hashlib.md5(data).digest()
    assert req.content_length == len(data)
    assert req.add_content_md5_metadata is True
    assert req.expires_in.ToTimedelta().total_seconds() == 60

    client.put.assert_awaited_once()
    put_call = client.put.await_args
    assert put_call[0][0] == "https://signed.example/put?sig=secret"
    sent_headers = put_call[1]["headers"]
    assert sent_headers["x-extra"] == "1"  # response headers are honored
    assert sent_headers["Content-Length"] == str(len(data))
    assert sent_headers["Content-MD5"] == b64encode(hashlib.md5(data).digest()).decode("utf-8")
    assert put_call[1]["content"] == data


@pytest.mark.asyncio
async def test_upload_traced_run_artifact_outputs_target_attempt():
    dataproxy = _make_dataproxy()
    _, ctx = _mock_http_client([httpx.Response(204)])

    with patch("httpx.AsyncClient", return_value=ctx):
        _, cluster = await upload_traced_run_artifact(
            dataproxy,
            kind="outputs",
            run_id=_RUN_ID,
            action_name="a0",
            attempt=2,
            data=b"outputs",
        )

    # Control-plane-served uploads report no routing cluster.
    assert cluster == ""
    req = dataproxy.create_traced_run_upload_location.await_args[0][0]
    assert req.filename_root == "traced-runs/local-x/a0/2"
    assert req.filename == "outputs.pb"


@pytest.mark.asyncio
async def test_upload_traced_run_artifact_report_filename():
    dataproxy = _make_dataproxy()
    _, ctx = _mock_http_client([httpx.Response(200)])

    with patch("httpx.AsyncClient", return_value=ctx):
        await upload_traced_run_artifact(
            dataproxy,
            kind="report",
            run_id=_RUN_ID,
            action_name="a1",
            attempt=1,
            data=b"<html/>",
            content_type="text/html",
        )

    req = dataproxy.create_traced_run_upload_location.await_args[0][0]
    assert req.filename_root == "traced-runs/local-x/a1/1"
    assert req.filename == "report.html"


@pytest.mark.asyncio
async def test_upload_traced_run_artifact_validates_kind_and_attempt():
    dataproxy = _make_dataproxy()
    # Unknown kind.
    with pytest.raises(ValueError, match="kind"):
        await upload_traced_run_artifact(
            dataproxy, kind="code", run_id=_RUN_ID, action_name="a0", attempt=None, data=b"x"
        )
    # Inputs must not carry an attempt.
    with pytest.raises(ValueError, match="attempt"):
        await upload_traced_run_artifact(
            dataproxy, kind="inputs", run_id=_RUN_ID, action_name="a0", attempt=1, data=b"x"
        )
    # Outputs / reports must carry an attempt.
    with pytest.raises(ValueError, match="attempt"):
        await upload_traced_run_artifact(
            dataproxy, kind="outputs", run_id=_RUN_ID, action_name="a0", attempt=None, data=b"x"
        )
    dataproxy.create_traced_run_upload_location.assert_not_awaited()


@pytest.mark.asyncio
async def test_upload_traced_run_artifact_maps_connect_errors():
    dataproxy = MagicMock()
    dataproxy.create_traced_run_upload_location = AsyncMock(
        side_effect=ConnectError(Code.PERMISSION_DENIED, "not yours")
    )

    with pytest.raises(RuntimeSystemError, match="not yours"):
        await upload_traced_run_artifact(
            dataproxy,
            kind="inputs",
            run_id=_RUN_ID,
            action_name="a0",
            attempt=None,
            data=b"x",
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
