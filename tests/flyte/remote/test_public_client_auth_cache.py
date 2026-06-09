from pathlib import Path
from unittest.mock import patch

import pytest

from flyte.remote._client.auth._public_client_cache import (
    CachedPublicClientAuthMetadata,
    fetch_public_client_auth_metadata,
    get_public_client_auth_metadata_cache_path,
    is_public_client_auth_metadata_cache_disabled,
    read_cached_public_client_auth_metadata,
    write_cached_public_client_auth_metadata,
)


def test_public_client_auth_metadata_cache_path_uses_per_org_domain_yaml_files(tmp_path: Path):
    path = get_public_client_auth_metadata_cache_path("dogfood.cloud-staging.union.ai", cache_root=tmp_path)
    assert path == tmp_path / "dogfood.cloud-staging.union.ai.yaml"


def test_public_client_auth_metadata_cache_round_trip(tmp_path: Path):
    metadata = CachedPublicClientAuthMetadata(
        authType="Pkce",
        clientId="dogfood-uctl",
        insecure=False,
        authorizationHeader="flyte-authorization",
        redirectUri="http://localhost:53593/callback",
        scopes=["all"],
        audience="dogfood-audience",
    )

    write_cached_public_client_auth_metadata("dogfood.cloud-staging.union.ai", metadata, cache_root=tmp_path)
    cached = read_cached_public_client_auth_metadata("dogfood.cloud-staging.union.ai", cache_root=tmp_path)

    assert cached == metadata


def test_public_client_auth_metadata_cache_write_failure_is_non_blocking(tmp_path: Path):
    metadata = CachedPublicClientAuthMetadata(
        authType="Pkce",
        clientId="dogfood-uctl",
        insecure=False,
        authorizationHeader="flyte-authorization",
        redirectUri="http://localhost:53593/callback",
        scopes=["all"],
        audience="dogfood-audience",
    )

    with patch("flyte.remote._client.auth._public_client_cache.Path.open", side_effect=OSError("nope")):
        assert (
            write_cached_public_client_auth_metadata("dogfood.cloud-staging.union.ai", metadata, cache_root=tmp_path)
            is None
        )


def test_public_client_auth_metadata_cache_can_be_disabled_with_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("FLYTE_AUTH_METADATA_CACHE_DISABLED", "true")
    metadata = CachedPublicClientAuthMetadata(
        authType="Pkce",
        clientId="dogfood-uctl",
        insecure=False,
        authorizationHeader="flyte-authorization",
        redirectUri="http://localhost:53593/callback",
        scopes=["all"],
        audience="dogfood-audience",
    )

    assert is_public_client_auth_metadata_cache_disabled() is True
    assert (
        write_cached_public_client_auth_metadata("dogfood.cloud-staging.union.ai", metadata, cache_root=tmp_path)
        is None
    )
    assert read_cached_public_client_auth_metadata("dogfood.cloud-staging.union.ai", cache_root=tmp_path) is None
    assert not get_public_client_auth_metadata_cache_path(
        "dogfood.cloud-staging.union.ai", cache_root=tmp_path
    ).exists()


@pytest.mark.asyncio
async def test_fetch_public_client_auth_metadata_surfaces_remote_errors():
    class FakeAuthClient:
        def __init__(self, address, http_client=None):
            self.address = address

        async def get_public_client_config(self, request):
            raise RuntimeError("boom")

    with (
        patch("flyte.remote._client.auth._public_client_cache.AuthMetadataServiceClient", FakeAuthClient),
        patch("flyte.remote._client.auth._session._resolve_tls_ca_cert", return_value=None),
        patch("flyte.remote._client.auth._session._build_pyqwest_client", return_value=object()),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            await fetch_public_client_auth_metadata("dogfood.cloud-staging.union.ai")
