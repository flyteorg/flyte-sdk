"""Tests for DeviceCodeAuthenticator config validation."""

from unittest.mock import AsyncMock, patch

import pytest

from flyte.remote._client.auth._authenticators.device_code import DeviceCodeAuthenticator
from flyte.remote._client.auth._client_config import ClientConfig
from flyte.remote._client.auth.errors import AuthenticationError


def _cfg(device_authorization_endpoint):
    return ClientConfig(
        token_endpoint="https://example.com/oauth2/token",
        authorization_endpoint="https://example.com/oauth2/authorize",
        redirect_uri="http://localhost:53593/callback",
        client_id="flytepropeller",
        scopes=["all"],
        device_authorization_endpoint=device_authorization_endpoint,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", [None, ""])
async def test_refresh_rejects_missing_device_authorization_endpoint(endpoint):
    """FLYTE-SDK-6P: an unadvertised device endpoint arrives as "" from the OAuth2
    metadata proto, not None. Both must raise the actionable AuthenticationError
    rather than falling through to httpx.post("")."""
    auth = DeviceCodeAuthenticator(endpoint="https://example.com")

    with (
        patch.object(DeviceCodeAuthenticator, "_resolve_config", AsyncMock(return_value=_cfg(endpoint))),
        patch(
            "flyte.remote._client.auth._token_client.get_device_code",
            AsyncMock(side_effect=AssertionError("must not be called")),
        ),
    ):
        with pytest.raises(AuthenticationError, match="Device Authentication is not available"):
            await auth._do_refresh_credentials()
