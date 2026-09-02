from __future__ import annotations

import pytest

from flyte.remote._client.auth._authenticators.base import Authenticator
from flyte.remote._client.auth._client_config import ClientConfig, StaticClientConfigStore
from flyte.remote._client.auth._keyring import Credentials


class _TestAuthenticator(Authenticator):
    async def _do_refresh_credentials(self) -> Credentials:
        raise NotImplementedError


def _remote_config() -> ClientConfig:
    return ClientConfig(
        token_endpoint="https://example.com/token",
        authorization_endpoint="https://example.com/authorize",
        redirect_uri="http://localhost:12345/callback",
        client_id="remote-client",
        scopes=["metadata-scope"],
    )


@pytest.mark.asyncio
async def test_configured_scopes_override_auth_metadata_scopes():
    authenticator = _TestAuthenticator(
        endpoint="https://example.com",
        cfg_store=StaticClientConfigStore(_remote_config()),
        scopes=["configured-scope-a", "configured-scope-b"],
        disable_keyring=True,
    )

    resolved = await authenticator._resolve_config()

    assert resolved.scopes == ["configured-scope-a", "configured-scope-b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("scopes", [None, []])
async def test_missing_configured_scopes_fall_back_to_auth_metadata(scopes):
    authenticator = _TestAuthenticator(
        endpoint="https://example.com",
        cfg_store=StaticClientConfigStore(_remote_config()),
        scopes=scopes,
        disable_keyring=True,
    )

    resolved = await authenticator._resolve_config()

    assert resolved.scopes == ["metadata-scope"]
