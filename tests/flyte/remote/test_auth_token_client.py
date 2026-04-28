from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from flyte.remote._client.auth._token_client import get_token


def _success_response(access_token: str = "access-token", refresh_token: str | None = None, expires_in: int = 3600):
    response = Mock(spec=httpx.Response)
    response.is_success = True
    payload = {
        "access_token": access_token,
        "expires_in": expires_in,
    }
    if refresh_token is not None:
        payload["refresh_token"] = refresh_token
    response.json.return_value = payload
    return response


@pytest.mark.asyncio
async def test_get_token_retries_transient_transport_errors(monkeypatch):
    session = Mock(spec=httpx.AsyncClient)
    session.post = AsyncMock(
        side_effect=[
            httpx.ConnectTimeout("connect timed out"),
            _success_response(refresh_token="refresh-token"),
        ]
    )
    sleep = AsyncMock()
    monkeypatch.setattr("flyte.remote._client.auth._token_client.asyncio.sleep", sleep)

    token, refresh_token, expires_in = await get_token(
        token_endpoint="https://issuer.example.com/oauth/token",
        http_session=session,
        client_id="client-id",
    )

    assert token == "access-token"
    assert refresh_token == "refresh-token"
    assert expires_in == 3600
    assert session.post.await_count == 2
    sleep.assert_awaited_once_with(0.5)


@pytest.mark.asyncio
async def test_get_token_raises_after_retry_budget_exhausted(monkeypatch):
    session = Mock(spec=httpx.AsyncClient)
    session.post = AsyncMock(side_effect=httpx.ConnectTimeout("connect timed out"))
    sleep = AsyncMock()
    monkeypatch.setattr("flyte.remote._client.auth._token_client.asyncio.sleep", sleep)

    with pytest.raises(httpx.ConnectTimeout):
        await get_token(
            token_endpoint="https://issuer.example.com/oauth/token",
            http_session=session,
            client_id="client-id",
        )

    assert session.post.await_count == 3
    assert sleep.await_count == 2
