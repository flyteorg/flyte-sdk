"""Tests for _upload_with_retry timeout and retry behavior."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from flyte.errors import RuntimeSystemError
from flyte.remote._data import _UPLOAD_TIMEOUT, _upload_with_retry


@pytest.fixture
def upload_file(tmp_path):
    f = tmp_path / "bundle.tar.gz"
    f.write_bytes(b"fake bundle content")
    return f


@pytest.mark.asyncio
async def test_upload_success(upload_file):
    resp = httpx.Response(200)
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.return_value = resp
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        result = await _upload_with_retry(upload_file, "https://signed.url/upload", {}, verify=True)
        assert result.status_code == 200
        mock_cls.assert_called_with(verify=True, timeout=_UPLOAD_TIMEOUT)


@pytest.mark.asyncio
async def test_upload_timeout_default():
    assert _UPLOAD_TIMEOUT.read == 600.0
    assert _UPLOAD_TIMEOUT.connect == 30.0


@pytest.mark.asyncio
async def test_upload_timeout_env_override():
    with patch.dict("os.environ", {"FLYTE_UPLOAD_TIMEOUT": "120"}):
        import importlib

        import flyte.remote._data as data_mod

        importlib.reload(data_mod)
        assert data_mod._UPLOAD_TIMEOUT.read == 120.0
        assert data_mod._UPLOAD_TIMEOUT.connect == 30.0

        # Restore default
        del data_mod
        import flyte.remote._data

        importlib.reload(flyte.remote._data)


@pytest.mark.asyncio
async def test_upload_retries_on_timeout(upload_file):
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.side_effect = httpx.ReadTimeout("timed out")
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with pytest.raises(RuntimeSystemError, match="timed out"):
            await _upload_with_retry(
                upload_file, "https://signed.url/upload", {}, verify=True, max_retries=2, min_backoff_sec=0.01
            )

        assert client.put.call_count == 3  # initial + 2 retries


@pytest.mark.asyncio
async def test_upload_retries_on_connect_error(upload_file):
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.side_effect = httpx.ConnectError("connection refused")
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with pytest.raises(RuntimeSystemError, match="connection refused"):
            await _upload_with_retry(
                upload_file, "https://signed.url/upload", {}, verify=True, max_retries=1, min_backoff_sec=0.01
            )

        assert client.put.call_count == 2


@pytest.mark.asyncio
async def test_upload_retries_on_server_error_then_succeeds(upload_file):
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.side_effect = [
            httpx.Response(503, text="service unavailable"),
            httpx.Response(200),
        ]
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        result = await _upload_with_retry(
            upload_file, "https://signed.url/upload", {}, verify=True, max_retries=3, min_backoff_sec=0.01
        )
        assert result.status_code == 200
        assert client.put.call_count == 2


@pytest.mark.asyncio
async def test_upload_no_retry_on_client_error(upload_file):
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.return_value = httpx.Response(403, text="forbidden")
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with pytest.raises(RuntimeSystemError, match="status 403"):
            await _upload_with_retry(
                upload_file, "https://signed.url/upload", {}, verify=True, max_retries=3, min_backoff_sec=0.01
            )

        assert client.put.call_count == 1  # no retries for 4xx
