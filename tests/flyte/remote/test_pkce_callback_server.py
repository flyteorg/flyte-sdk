"""
Tests for the local OAuth2 callback server used by the PKCE (browser) login flow.

The redirect URI is supplied by the deployment's public client config. When it is
missing or has no host:port, `asyncio.start_server(None, None)` used to raise a bare
`ValueError: Neither host/port nor sock were specified`, which reads like an SDK bug
rather than the configuration problem it is.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flyte.errors import InitializationError
from flyte.remote._client.auth._authenticators.pkce import AuthorizationClient


def _client(redirect_uri) -> AuthorizationClient:
    return AuthorizationClient(
        endpoint="dns:///example.com",
        auth_endpoint="https://example.com/oauth2/authorize",
        token_endpoint="https://example.com/oauth2/token",
        http_session=MagicMock(),
        client_id="flytectl",
        redirect_uri=redirect_uri,
    )


class TestCallbackServerRedirectUriValidation:
    @pytest.mark.parametrize(
        "redirect_uri",
        [
            pytest.param(None, id="missing"),
            pytest.param("", id="empty"),
            pytest.param("localhost:8080/callback", id="no-scheme"),  # urlparse reads "localhost" as the scheme
            pytest.param("http://localhost/callback", id="no-port"),
            pytest.param("/callback", id="path-only"),
        ],
    )
    @pytest.mark.asyncio
    async def test_unusable_redirect_uri_raises_initialization_error(self, redirect_uri):
        client = _client(redirect_uri)

        with patch("asyncio.start_server", new_callable=AsyncMock) as mock_start_server:
            with pytest.raises(InitializationError) as exc_info:
                await client._create_callback_server()

        # We fail before touching the event loop, so no half-bound server is left behind.
        mock_start_server.assert_not_called()

        err = exc_info.value
        assert err.code == "InvalidRedirectURI"
        assert err.kind == "user"
        # The offending value is named so the user knows what to fix.
        assert repr(redirect_uri) in str(err)

    @pytest.mark.asyncio
    async def test_valid_redirect_uri_binds_that_host_and_port(self):
        client = _client("http://localhost:8080/callback")

        with patch("asyncio.start_server", new_callable=AsyncMock) as mock_start_server:
            server, _queue, handler = await client._create_callback_server()

        assert server is mock_start_server.return_value
        _handle, host, port = mock_start_server.call_args.args
        assert (host, port) == ("localhost", 8080)
        # The callback handler matches incoming requests on the redirect URI's path.
        assert handler.redirect_path == "/callback"
        mock_start_server.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_loopback_redirect_uri_is_still_accepted(self):
        """Only host/port presence is validated -- we do not second-guess the deployment."""
        client = _client("https://127.0.0.1:53593/oauth2/callback")

        with patch("asyncio.start_server", new_callable=AsyncMock) as mock_start_server:
            await client._create_callback_server()

        _handle, host, port = mock_start_server.call_args.args
        assert (host, port) == ("127.0.0.1", 53593)
