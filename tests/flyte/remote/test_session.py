import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from flyte.remote._client.auth._session import create_session, SessionConfig


@pytest.mark.asyncio
@patch("flyte.remote._client.auth._session.create_auth_interceptors", return_value=[])
@patch("flyte.remote._client.auth._session.create_proxy_auth_interceptors", return_value=[])
@patch("flyte.remote._client.auth._session.get_async_session")
async def test_insecure_skips_auth_interceptors(mock_get_session, mock_proxy, mock_auth):
    mock_get_session.return_value = MagicMock()
    result = await create_session("localhost:8080", insecure=True)
    assert isinstance(result, SessionConfig)
    mock_auth.assert_not_called()
    assert result.endpoint == "http://localhost:8080"


@pytest.mark.asyncio
@patch("flyte.remote._client.auth._session.create_auth_interceptors", return_value=["auth1"])
@patch("flyte.remote._client.auth._session.create_proxy_auth_interceptors", return_value=[])
@patch("flyte.remote._client.auth._session.get_async_session")
async def test_secure_creates_auth_interceptors(mock_get_session, mock_proxy, mock_auth):
    mock_get_session.return_value = MagicMock()
    result = await create_session("example.com:443", insecure=False)
    mock_auth.assert_called_once()
    assert "auth1" in result.interceptors
    assert result.endpoint == "https://example.com:443"


@pytest.mark.asyncio
@patch("flyte.remote._client.auth._session.create_auth_interceptors", return_value=[])
@patch("flyte.remote._client.auth._session.create_proxy_auth_interceptors", return_value=[])
@patch("flyte.remote._client.auth._session.get_async_session")
async def test_rpc_retries_creates_retry_interceptors(mock_get_session, mock_proxy, mock_auth):
    mock_get_session.return_value = MagicMock()
    result = await create_session("example.com:443", insecure=True, rpc_retries=3)
    # Should have: DefaultMetadataInterceptor + RetryUnaryInterceptor + RetryServerStreamInterceptor
    from flyte.remote._client.auth._interceptors.retry import RetryUnaryInterceptor, RetryServerStreamInterceptor

    retry_unary = [i for i in result.interceptors if isinstance(i, RetryUnaryInterceptor)]
    retry_stream = [i for i in result.interceptors if isinstance(i, RetryServerStreamInterceptor)]
    assert len(retry_unary) == 1
    assert len(retry_stream) == 1
    assert retry_unary[0]._max_attempts == 4  # rpc_retries + 1


@pytest.mark.asyncio
@patch("flyte.remote._client.auth._session.create_auth_interceptors", return_value=[])
@patch("flyte.remote._client.auth._session.create_proxy_auth_interceptors", return_value=[])
@patch("flyte.remote._client.auth._session.get_async_session")
async def test_returns_session_config(mock_get_session, mock_proxy, mock_auth):
    mock_http = MagicMock()
    mock_get_session.return_value = mock_http
    result = await create_session("dns:///example.com:8089", insecure=False)
    assert isinstance(result, SessionConfig)
    assert result.endpoint == "https://example.com:8089"
    assert result.http_client is mock_http
    assert len(result.interceptors) >= 1  # At least DefaultMetadataInterceptor
