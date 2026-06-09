from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from flyte.remote._client.auth._client_config import LocalClientConfigOverrides, RemoteClientConfigStore
from flyte.remote._client.auth._public_client_cache import (
    CachedPublicClientAuthMetadata,
    get_public_client_auth_metadata_cache_path,
    read_cached_public_client_auth_metadata,
    write_cached_public_client_auth_metadata,
)


@pytest.mark.asyncio
async def test_remote_client_config_store_skips_public_client_config_when_endpoint_cache_exists(tmp_path: Path):
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

    store = RemoteClientConfigStore(
        "https://dogfood.cloud-staging.union.ai",
        cache_root=tmp_path,
        client_config_overrides=LocalClientConfigOverrides(audience="my-audience"),
    )
    store._client = Mock()
    store._client.get_o_auth2_metadata = AsyncMock(
        return_value=SimpleNamespace(
            token_endpoint="https://example.com/token",
            authorization_endpoint="https://example.com/authorize",
            device_authorization_endpoint="https://example.com/device",
        )
    )
    store._client.get_public_client_config = AsyncMock()

    cfg = await store.get_client_config()

    store._client.get_o_auth2_metadata.assert_awaited_once()
    store._client.get_public_client_config.assert_not_called()
    assert cfg.client_id == "cached-client-id"
    assert cfg.redirect_uri == "http://localhost:53593/callback"
    assert cfg.scopes == ["all"]
    assert cfg.header_key == "flyte-authorization"
    assert cfg.audience == "my-audience"


@pytest.mark.asyncio
async def test_remote_client_config_store_fetches_public_client_config_when_endpoint_cache_is_missing(tmp_path: Path):
    store = RemoteClientConfigStore(
        "https://dogfood.cloud-staging.union.ai",
        auth_type="Pkce",
        cache_root=tmp_path,
    )
    store._client = Mock()
    store._client.get_o_auth2_metadata = AsyncMock(
        return_value=SimpleNamespace(
            token_endpoint="https://example.com/token",
            authorization_endpoint="https://example.com/authorize",
            device_authorization_endpoint="https://example.com/device",
        )
    )
    store._client.get_public_client_config = AsyncMock(
        return_value=SimpleNamespace(
            redirect_uri="http://localhost:53593/callback",
            client_id="remote-client-id",
            scopes=["scope-a"],
            authorization_metadata_key="flyte-authorization",
            audience="remote-audience",
        )
    )

    cfg = await store.get_client_config()

    store._client.get_o_auth2_metadata.assert_awaited_once()
    store._client.get_public_client_config.assert_awaited_once()
    assert cfg.client_id == "remote-client-id"
    assert cfg.redirect_uri == "http://localhost:53593/callback"
    assert cfg.scopes == ["scope-a"]
    assert cfg.header_key == "flyte-authorization"
    assert cfg.audience == "remote-audience"

    cached = read_cached_public_client_auth_metadata("dogfood.cloud-staging.union.ai", cache_root=tmp_path)
    assert cached is not None
    assert cached.client_id == "remote-client-id"
    assert cached.scopes == ["scope-a"]
    assert cached.audience == "remote-audience"


@pytest.mark.asyncio
async def test_remote_client_config_store_treats_malformed_endpoint_cache_as_miss(tmp_path: Path):
    cache_path = get_public_client_auth_metadata_cache_path(
        "dogfood.cloud-staging.union.ai", cache_root=tmp_path
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("not: [valid")

    store = RemoteClientConfigStore(
        "https://dogfood.cloud-staging.union.ai",
        auth_type="Pkce",
        cache_root=tmp_path,
    )
    store._client = Mock()
    store._client.get_o_auth2_metadata = AsyncMock(
        return_value=SimpleNamespace(
            token_endpoint="https://example.com/token",
            authorization_endpoint="https://example.com/authorize",
            device_authorization_endpoint="https://example.com/device",
        )
    )
    store._client.get_public_client_config = AsyncMock(
        return_value=SimpleNamespace(
            redirect_uri="http://localhost:53593/callback",
            client_id="remote-client-id",
            scopes=["scope-a"],
            authorization_metadata_key="flyte-authorization",
            audience="remote-audience",
        )
    )

    cfg = await store.get_client_config()

    store._client.get_public_client_config.assert_awaited_once()
    assert cfg.client_id == "remote-client-id"


@pytest.mark.asyncio
async def test_remote_client_config_store_uses_endpoint_specific_cache_files(tmp_path: Path):
    write_cached_public_client_auth_metadata(
        "other-org.cloud-staging.union.ai",
        CachedPublicClientAuthMetadata(
            authType="Pkce",
            clientId="other-client-id",
            insecure=False,
            authorizationHeader="flyte-authorization",
            redirectUri="http://localhost:53593/callback",
            scopes=["all"],
            audience="other-audience",
        ),
        cache_root=tmp_path,
    )

    store = RemoteClientConfigStore(
        "https://dogfood.cloud-staging.union.ai",
        auth_type="Pkce",
        cache_root=tmp_path,
    )
    store._client = Mock()
    store._client.get_o_auth2_metadata = AsyncMock(
        return_value=SimpleNamespace(
            token_endpoint="https://example.com/token",
            authorization_endpoint="https://example.com/authorize",
            device_authorization_endpoint="https://example.com/device",
        )
    )
    store._client.get_public_client_config = AsyncMock(
        return_value=SimpleNamespace(
            redirect_uri="http://localhost:53593/callback",
            client_id="dogfood-client-id",
            scopes=["scope-a"],
            authorization_metadata_key="flyte-authorization",
            audience="remote-audience",
        )
    )

    cfg = await store.get_client_config()

    store._client.get_public_client_config.assert_awaited_once()
    assert cfg.client_id == "dogfood-client-id"


@pytest.mark.asyncio
async def test_remote_client_config_store_accepts_cached_metadata_without_audience(tmp_path: Path):
    cache_path = get_public_client_auth_metadata_cache_path(
        "dogfood.cloud-staging.union.ai", cache_root=tmp_path
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        """authType: Pkce
clientId: cached-client-id
insecure: false
authorizationHeader: flyte-authorization
redirectUri: http://localhost:53593/callback
scopes:
  - all
"""
    )

    store = RemoteClientConfigStore(
        "https://dogfood.cloud-staging.union.ai",
        cache_root=tmp_path,
    )
    store._client = Mock()
    store._client.get_o_auth2_metadata = AsyncMock(
        return_value=SimpleNamespace(
            token_endpoint="https://example.com/token",
            authorization_endpoint="https://example.com/authorize",
            device_authorization_endpoint="https://example.com/device",
        )
    )
    store._client.get_public_client_config = AsyncMock()

    cfg = await store.get_client_config()

    store._client.get_o_auth2_metadata.assert_awaited_once()
    store._client.get_public_client_config.assert_not_called()
    assert cfg.client_id == "cached-client-id"
    assert cfg.redirect_uri == "http://localhost:53593/callback"
    assert cfg.scopes == ["all"]
    assert cfg.header_key == "flyte-authorization"
    assert cfg.audience is None
