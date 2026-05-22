from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from flyte.remote._client.auth._client_config import ClientConfig, RemoteClientConfigStore


@pytest.mark.asyncio
async def test_remote_client_config_store_skips_public_client_config_when_local_public_fields_are_complete():
    local_client_config = ClientConfig(
        client_id="client-id",
        scopes=["scope-a"],
        header_key="flyte-authorization",
        redirect_uri="http://localhost:53593/callback",
        audience="my-audience",
    )
    store = RemoteClientConfigStore("https://example.com", client_config=local_client_config)
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
    assert cfg.client_id == "client-id"
    assert cfg.redirect_uri == "http://localhost:53593/callback"
    assert cfg.scopes == ["scope-a"]
    assert cfg.header_key == "flyte-authorization"
    assert cfg.audience == "my-audience"
    assert cfg.token_endpoint == "https://example.com/token"
    assert cfg.authorization_endpoint == "https://example.com/authorize"


@pytest.mark.asyncio
async def test_remote_client_config_store_fetches_public_client_config_when_local_public_fields_are_incomplete():
    local_client_config = ClientConfig(client_id="client-id")
    store = RemoteClientConfigStore("https://example.com", client_config=local_client_config)
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
