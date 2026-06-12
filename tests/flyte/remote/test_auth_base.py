from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from flyte.remote._client.auth._authenticators.base import Authenticator
from flyte.remote._client.auth._client_config import RemoteClientConfigStore
from flyte.remote._client.auth._keyring import Credentials
from flyte.remote._client.auth._public_client_cache import (
    CachedPublicClientAuthMetadata,
    read_cached_public_client_auth_metadata,
    write_cached_public_client_auth_metadata,
)


class RetryingTestAuthenticator(Authenticator):
    async def _do_refresh_credentials(self) -> Credentials:
        cfg = await self._resolve_config()
        if cfg.client_id == "cached-client-id":
            raise RuntimeError("stale cache")
        return Credentials(access_token=f"token-for-{cfg.client_id}", for_endpoint=self._endpoint)


@pytest.mark.asyncio
async def test_refresh_credentials_invalidates_stale_cache_and_retries(tmp_path):
    write_cached_public_client_auth_metadata(
        "dogfood.cloud-staging.union.ai",
        CachedPublicClientAuthMetadata(
            authType="Pkce",
            clientId="cached-client-id",
            insecure=False,
            authorizationHeader="flyte-authorization",
            redirectUri="http://localhost:53593/callback",
            scopes=["all"],
            audience="cached-audience",
        ),
        cache_root=tmp_path,
    )

    cfg_store = RemoteClientConfigStore(
        "https://dogfood.cloud-staging.union.ai",
        auth_type="Pkce",
        cache_root=tmp_path,
    )
    cfg_store._client = Mock()
    cfg_store._client.get_o_auth2_metadata = AsyncMock(
        return_value=SimpleNamespace(
            token_endpoint="https://example.com/token",
            authorization_endpoint="https://example.com/authorize",
            device_authorization_endpoint="https://example.com/device",
        )
    )
    cfg_store._client.get_public_client_config = AsyncMock(
        return_value=SimpleNamespace(
            redirect_uri="http://localhost:53593/callback",
            client_id="fresh-client-id",
            scopes=["scope-a"],
            authorization_metadata_key="flyte-authorization",
            audience="fresh-audience",
        )
    )

    authenticator = RetryingTestAuthenticator(
        endpoint="https://dogfood.cloud-staging.union.ai",
        cfg_store=cfg_store,
        disable_keyring=True,
    )

    await authenticator.refresh_credentials()

    assert cfg_store._client.get_o_auth2_metadata.await_count == 2
    cfg_store._client.get_public_client_config.assert_awaited_once()
    creds = authenticator.get_credentials()
    assert creds is not None
    assert creds.access_token == "token-for-fresh-client-id"

    cached = read_cached_public_client_auth_metadata("dogfood.cloud-staging.union.ai", cache_root=tmp_path)
    assert cached is not None
    assert cached.client_id == "fresh-client-id"
    assert cached.audience == "fresh-audience"
