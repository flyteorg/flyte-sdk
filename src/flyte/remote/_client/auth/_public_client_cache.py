from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pydantic
import rich_click as click
import yaml
from flyteidl2.auth.auth_service_connect import AuthMetadataServiceClient
from flyteidl2.auth.auth_service_pb2 import GetPublicClientConfigRequest

from flyte._logging import logger
from flyte._utils import sanitize_endpoint
from flyte.syncify import syncify

_AUTH_METADATA_CACHE_DISABLED_ENV = "FLYTE_AUTH_METADATA_CACHE_DISABLED"
_TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


class CachedPublicClientAuthMetadata(pydantic.BaseModel):
    auth_type: str = pydantic.Field(alias="authType")
    client_id: str = pydantic.Field(alias="clientId")
    insecure: bool
    authorization_header: str = pydantic.Field(alias="authorizationHeader")
    redirect_uri: str = pydantic.Field(alias="redirectUri")
    scopes: list[str]
    audience: str | None = None

    model_config = pydantic.ConfigDict(populate_by_name=True)

    def to_yaml_dict(self) -> dict:
        return self.model_dump(by_alias=True)

    def to_admin_dict(self) -> dict[str, object]:
        return self.to_yaml_dict()

    def to_local_client_config_overrides_kwargs(self) -> dict[str, object]:
        return {
            "client_id": self.client_id,
            "header_key": self.authorization_header,
            "redirect_uri": self.redirect_uri,
            "scopes": list(self.scopes),
            "audience": self.audience,
        }

    @classmethod
    def from_public_client_config(
        cls,
        public_client_config,
        *,
        auth_type: str = "Pkce",
        insecure: bool = False,
    ) -> "CachedPublicClientAuthMetadata":
        return cls(
            authType=auth_type,
            clientId=public_client_config.client_id,
            insecure=insecure,
            authorizationHeader=public_client_config.authorization_metadata_key,
            redirectUri=public_client_config.redirect_uri,
            scopes=list(public_client_config.scopes),
            audience=public_client_config.audience,
        )


def get_public_client_auth_metadata_cache_root(cache_root: Path | None = None) -> Path:
    return cache_root or Path.home() / ".flyte" / ".cache"


def is_public_client_auth_metadata_cache_disabled() -> bool:
    return os.environ.get(_AUTH_METADATA_CACHE_DISABLED_ENV, "").lower() in _TRUE_ENV_VALUES


def extract_public_client_auth_metadata_cache_key(endpoint: str) -> str:
    sanitized = sanitize_endpoint(endpoint) or endpoint
    return sanitized.removeprefix("dns:///").removeprefix("dns://").lstrip("/")


def get_public_client_auth_metadata_cache_path(endpoint: str, cache_root: Path | None = None) -> Path:
    resolved_root = get_public_client_auth_metadata_cache_root(cache_root)
    return resolved_root / f"{extract_public_client_auth_metadata_cache_key(endpoint)}.yaml"


def read_cached_public_client_auth_metadata(
    endpoint: str, cache_root: Path | None = None
) -> Optional[CachedPublicClientAuthMetadata]:
    if is_public_client_auth_metadata_cache_disabled():
        return None

    file_path = get_public_client_auth_metadata_cache_path(endpoint, cache_root=cache_root)
    if not file_path.exists():
        return None

    try:
        with file_path.open() as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid cached auth metadata content in {file_path}")
        return CachedPublicClientAuthMetadata.model_validate(payload)
    except (OSError, yaml.YAMLError, pydantic.ValidationError, ValueError) as e:
        logger.warning(f"Failed to read cached auth metadata from {file_path}: {e}")
        return None


def write_cached_public_client_auth_metadata(
    endpoint: str,
    metadata: CachedPublicClientAuthMetadata,
    cache_root: Path | None = None,
) -> Path | None:
    if is_public_client_auth_metadata_cache_disabled():
        return None

    file_path = get_public_client_auth_metadata_cache_path(endpoint, cache_root=cache_root)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w") as handle:
            yaml.safe_dump(metadata.to_yaml_dict(), handle, sort_keys=False)
        click.echo(f"Auth metadata saved at {file_path}")
        return file_path
    except OSError as e:
        logger.warning(f"Failed to write cached auth metadata to {file_path}: {e}")
        return None


async def fetch_public_client_auth_metadata(
    endpoint: str,
    *,
    auth_type: str = "Pkce",
    insecure: bool = False,
    insecure_skip_verify: bool = False,
    ca_cert_file_path: str | None = None,
) -> CachedPublicClientAuthMetadata:
    from flyte.remote._client.auth._session import (
        _build_pyqwest_client,
        _resolve_tls_ca_cert,
        normalize_rpc_endpoint,
    )

    rpc_endpoint = normalize_rpc_endpoint(endpoint, insecure=insecure)
    tls_ca_cert = await _resolve_tls_ca_cert(
        rpc_endpoint,
        insecure=insecure,
        insecure_skip_verify=insecure_skip_verify,
        ca_cert_file_path=ca_cert_file_path,
    )
    client = AuthMetadataServiceClient(address=rpc_endpoint, http_client=_build_pyqwest_client(tls_ca_cert))
    public_client_config = await client.get_public_client_config(GetPublicClientConfigRequest())
    return CachedPublicClientAuthMetadata.from_public_client_config(
        public_client_config,
        auth_type=auth_type,
        insecure=insecure,
    )


@syncify
async def fetch_public_client_auth_metadata_sync(
    endpoint: str,
    *,
    auth_type: str = "Pkce",
    insecure: bool = False,
    insecure_skip_verify: bool = False,
    ca_cert_file_path: str | None = None,
) -> CachedPublicClientAuthMetadata:
    return await fetch_public_client_auth_metadata(
        endpoint,
        auth_type=auth_type,
        insecure=insecure,
        insecure_skip_verify=insecure_skip_verify,
        ca_cert_file_path=ca_cert_file_path,
    )
