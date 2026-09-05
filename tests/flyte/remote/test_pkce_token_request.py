"""
Tests for the authorization-code -> access-token exchange in the PKCE (browser) login flow.

A token endpoint that rejects the exchange is telling us the deployment's OAuth2
application is misconfigured -- "client_secret is missing.", a redirect_uri mismatch, an
unknown client. That used to surface as a bare `RuntimeError` carrying the raw response
bytes, which reads as an SDK crash (FLYTE-SDK-7D) and got reported to Sentry as one.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from flyte.remote._client.auth._authenticators.pkce import AuthorizationClient
from flyte.remote._client.auth.errors import AuthenticationError


def _client(response) -> AuthorizationClient:
    session = MagicMock()
    session.post = AsyncMock(return_value=response)
    return AuthorizationClient(
        endpoint="dns:///example.union.ai",
        auth_endpoint="https://example.union.ai/oauth2/authorize",
        token_endpoint="https://example.union.ai/oauth2/token",
        http_session=session,
        client_id="flytepropeller",
        redirect_uri="http://localhost:8080/callback",
    )


def _response(status_code: int, *, json_body=None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_body is None:
        resp.json.side_effect = ValueError("not json")
    else:
        resp.json.return_value = json_body
    return resp


def _auth_code(state: str) -> SimpleNamespace:
    return SimpleNamespace(code="the-auth-code", state=state)


@pytest.mark.asyncio
async def test_rejected_token_request_raises_authentication_error():
    """The exact FLYTE-SDK-7D shape: 400 invalid_request / "client_secret is missing."."""
    client = _client(
        _response(
            400,
            json_body={"error": "invalid_request", "error_description": "client_secret is missing."},
            text='{"error": "invalid_request", "error_description": "client_secret is missing."}',
        )
    )

    with pytest.raises(AuthenticationError) as exc_info:
        await client._request_access_token(_auth_code(client._state))

    message = str(exc_info.value)
    # The IDP's own description is the only part that says *why*, so it has to survive.
    assert "client_secret is missing." in message
    assert "invalid_request" in message
    assert "400" in message
    assert "https://example.union.ai/oauth2/token" in message


@pytest.mark.asyncio
async def test_rejected_token_request_is_filtered_from_sentry():
    """AuthenticationError is on _sentry's user-error list; a bare RuntimeError was not."""
    from flyte._sentry import _is_user_error

    client = _client(_response(401, json_body={"error": "invalid_client"}, text='{"error": "invalid_client"}'))

    with pytest.raises(AuthenticationError) as exc_info:
        await client._request_access_token(_auth_code(client._state))

    assert _is_user_error(exc_info.value)


@pytest.mark.asyncio
async def test_rejected_token_request_with_non_json_body_quotes_the_body():
    """A proxy or login page answering the token endpoint returns HTML, not RFC 6749 JSON."""
    client = _client(_response(502, text="<html><body>502 Bad Gateway</body></html>"))

    with pytest.raises(AuthenticationError) as exc_info:
        await client._request_access_token(_auth_code(client._state))

    assert "502 Bad Gateway" in str(exc_info.value)


@pytest.mark.asyncio
async def test_rejected_token_request_truncates_a_huge_body():
    client = _client(_response(500, text="x" * 5000))

    with pytest.raises(AuthenticationError) as exc_info:
        await client._request_access_token(_auth_code(client._state))

    assert len(str(exc_info.value)) < 1000
    assert "..." in str(exc_info.value)


@pytest.mark.asyncio
async def test_rejected_token_request_with_empty_body_still_explains_itself():
    client = _client(_response(403, text=""))

    with pytest.raises(AuthenticationError) as exc_info:
        await client._request_access_token(_auth_code(client._state))

    assert "the response body was empty" in str(exc_info.value)
    assert "403" in str(exc_info.value)


@pytest.mark.asyncio
async def test_successful_token_request_still_returns_credentials():
    client = _client(
        _response(
            200,
            json_body={"access_token": "at", "refresh_token": "rt", "expires_in": 3600},
            text="",
        )
    )

    creds = await client._request_access_token(_auth_code(client._state))

    assert creds.access_token == "at"
    assert creds.refresh_token == "rt"


@pytest.mark.asyncio
async def test_response_without_access_token_raises_authentication_error():
    client = _client(_response(200, json_body={"token_type": "Bearer"}, text="{}"))

    with pytest.raises(AuthenticationError) as exc_info:
        await client._request_access_token(_auth_code(client._state))

    assert "access_token" in str(exc_info.value)


@pytest.mark.asyncio
async def test_state_mismatch_still_raises_value_error():
    """The state check guards against a forged callback and is deliberately left alone."""
    client = _client(_response(200, json_body={"access_token": "at"}, text=""))

    with pytest.raises(ValueError, match="Unexpected state parameter"):
        await client._request_access_token(_auth_code("not-the-state-we-sent"))
