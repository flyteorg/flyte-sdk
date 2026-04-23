"""Tests for admin.apiKeyEnvVar platform configuration."""

from __future__ import annotations

import base64

import pytest
import yaml

from flyte.config._config import Config, PlatformConfig


def _encode_api_key(endpoint: str, client_id: str, client_secret: str, org: str) -> str:
    raw = f"{endpoint}:{client_id}:{client_secret}:{org}"
    return base64.b64encode(raw.encode("utf-8")).decode("utf-8")


@pytest.fixture
def encoded_key() -> str:
    return _encode_api_key(
        endpoint="cfg-test.union.ai",
        client_id="cfg-client",
        client_secret="cfg-secret",
        org="cfg-org",
    )


def test_platform_config_loads_api_key_from_custom_env_var(tmp_path, monkeypatch, encoded_key: str) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "admin": {
                    "apiKeyEnvVar": "MY_FLYTE_API_KEY",
                    "authType": "Pkce",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MY_FLYTE_API_KEY", encoded_key)

    cfg = Config.auto(cfg_path)

    assert cfg.platform.api_key == encoded_key
    assert cfg.platform.auth_mode == "ClientSecret"
    assert cfg.platform.client_id == "cfg-client"
    assert cfg.platform.client_credentials_secret == "cfg-secret"
    assert cfg.platform.endpoint == "cfg-test.union.ai"


def test_platform_config_repr_redacts_api_key(encoded_key: str) -> None:
    text = repr(PlatformConfig(api_key=encoded_key))
    assert "***" in text
    assert encoded_key not in text


def test_platform_config_repr_redacts_client_secret() -> None:
    text = repr(PlatformConfig(client_credentials_secret="super-secret"))
    assert "***" in text
    assert "super-secret" not in text


def test_platform_config_api_key_env_var_missing_key(tmp_path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.dump({"admin": {"apiKeyEnvVar": "UNSET_FLYTE_KEY_VAR"}}),
        encoding="utf-8",
    )
    monkeypatch.delenv("UNSET_FLYTE_KEY_VAR", raising=False)

    cfg = Config.auto(cfg_path)

    assert cfg.platform.api_key is None
