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
async def test_upload_retries_on_read_error(upload_file):
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.side_effect = httpx.ReadError("connection reset by peer")
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with pytest.raises(RuntimeSystemError, match="connection reset by peer"):
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


@pytest.mark.asyncio
async def test_upload_honors_retry_after_seconds(upload_file):
    """When the server returns 429 with Retry-After: <int>, we sleep that long."""
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.side_effect = [
            httpx.Response(429, headers={"Retry-After": "2"}, text="slow down"),
            httpx.Response(200),
        ]
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with patch("flyte.remote._data.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            result = await _upload_with_retry(
                upload_file, "https://signed.url/upload", {}, verify=True, max_retries=3, min_backoff_sec=0.01
            )

        assert result.status_code == 200
        assert client.put.call_count == 2
        # Honored the Retry-After value (2s) rather than the exponential value (~0.01s).
        mock_sleep.assert_awaited_once_with(2.0)


@pytest.mark.asyncio
async def test_upload_caps_absurd_retry_after(upload_file):
    """A misbehaving server returning Retry-After: 99999 should be clamped."""
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.side_effect = [
            httpx.Response(429, headers={"Retry-After": "99999"}, text="slow down"),
            httpx.Response(200),
        ]
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with patch("flyte.remote._data.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            result = await _upload_with_retry(
                upload_file,
                "https://signed.url/upload",
                {},
                verify=True,
                max_retries=3,
                min_backoff_sec=0.01,
                retry_after_cap_sec=5.0,
            )

        assert result.status_code == 200
        mock_sleep.assert_awaited_once_with(5.0)


@pytest.mark.asyncio
async def test_upload_429_without_retry_after_uses_exponential(upload_file):
    """If no Retry-After header is sent, normal exponential backoff applies."""
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.side_effect = [
            httpx.Response(429, text="slow down"),
            httpx.Response(200),
        ]
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with patch("flyte.remote._data.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            result = await _upload_with_retry(
                upload_file,
                "https://signed.url/upload",
                {},
                verify=True,
                max_retries=3,
                min_backoff_sec=0.01,
                max_backoff_sec=10.0,
            )

        assert result.status_code == 200
        # Exponential value for the first retry: min_backoff_sec * 2**0 = 0.01
        mock_sleep.assert_awaited_once_with(0.01)
