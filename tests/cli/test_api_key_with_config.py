"""Tests that FLYTE_API_KEY composes with config file settings."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import click

from flyte.cli._common import CLIConfig
from flyte.config._config import Config, ImageConfig, PlatformConfig


def _encode_api_key(endpoint: str, client_id: str, client_secret: str, org: str) -> str:
    raw = f"{endpoint}:{client_id}:{client_secret}:{org}"
    return base64.b64encode(raw.encode("utf-8")).decode("utf-8")


API_KEY = _encode_api_key(
    endpoint="my-union.cloud.union.ai",
    client_id="my-client-id",
    client_secret="my-secret",
    org="my-org",
)


def _make_cli_config(
    config: Config | None = None,
    endpoint: str | None = None,
    org: str | None = None,
) -> CLIConfig:
    ctx = MagicMock(spec=click.Context)
    return CLIConfig(
        config=config or Config(),
        ctx=ctx,
        endpoint=endpoint,
        org=org,
    )


class TestApiKeyWithConfig:
    """Verify that FLYTE_API_KEY and config file settings compose correctly."""

    @patch.dict("os.environ", {"FLYTE_API_KEY": API_KEY})
    @patch("flyte.cli._common.flyte")
    def test_api_key_with_config_file_preserves_image_builder(self, mock_flyte):
        """When both FLYTE_API_KEY and a config file with image.builder are present,
        init_from_config should be called with API key auth AND the image settings."""
        cfg = Config(
            image=ImageConfig(builder="remote"),
            source="/fake/.flyte/config.yaml",
        )
        cli = _make_cli_config(config=cfg)
        cli.init()

        mock_flyte.init_from_config.assert_called_once()
        call_cfg = mock_flyte.init_from_config.call_args[0][0]

        # Auth fields from API key
        assert call_cfg.platform.client_id == "my-client-id"
        assert call_cfg.platform.client_credentials_secret == "my-secret"
        assert call_cfg.platform.auth_mode == "ClientSecret"
        assert call_cfg.platform.endpoint is not None

        # Image settings from config file preserved
        assert call_cfg.image.builder == "remote"

        # Org from API key
        assert call_cfg.task.org == "my-org"

        # init_from_api_key should NOT be called
        mock_flyte.init_from_api_key.assert_not_called()

    @patch.dict("os.environ", {"FLYTE_API_KEY": API_KEY})
    @patch("flyte.cli._common.flyte")
    def test_api_key_without_config_file(self, mock_flyte):
        """When FLYTE_API_KEY is set and no config file exists,
        init_from_config should still be called (not init_from_api_key)."""
        cli = _make_cli_config()
        cli.init()

        mock_flyte.init_from_config.assert_called_once()
        call_cfg = mock_flyte.init_from_config.call_args[0][0]

        assert call_cfg.platform.client_id == "my-client-id"
        assert call_cfg.platform.client_credentials_secret == "my-secret"
        assert call_cfg.platform.auth_mode == "ClientSecret"
        mock_flyte.init_from_api_key.assert_not_called()

    @patch.dict("os.environ", {"FLYTE_API_KEY": API_KEY})
    @patch("flyte.cli._common.flyte")
    def test_cli_endpoint_overrides_api_key_endpoint(self, mock_flyte):
        """CLI --endpoint flag should take precedence over endpoint from API key."""
        cli = _make_cli_config(endpoint="cli-override.cloud.union.ai")
        cli.init()

        mock_flyte.init_from_config.assert_called_once()
        call_cfg = mock_flyte.init_from_config.call_args[0][0]

        # CLI endpoint wins over the API key's endpoint
        assert call_cfg.platform.endpoint == "cli-override.cloud.union.ai"

    @patch.dict("os.environ", {"FLYTE_API_KEY": API_KEY})
    @patch("flyte.cli._common.flyte")
    def test_cli_org_overrides_api_key_org(self, mock_flyte):
        """CLI --org flag should take precedence over org from API key."""
        cli = _make_cli_config(org="cli-org")
        cli.init()

        mock_flyte.init_from_config.assert_called_once()
        call_cfg = mock_flyte.init_from_config.call_args[0][0]

        assert call_cfg.task.org == "cli-org"

    @patch.dict(
        "os.environ",
        {
            "FLYTE_API_KEY": _encode_api_key(
                "ep.union.ai",
                "cid",
                "cs",
                "",
            )
        },
    )
    @patch("flyte.cli._common.flyte")
    def test_api_key_with_none_org_does_not_set_org(self, mock_flyte):
        """When API key org is literal 'None', task org should not be overridden."""
        cli = _make_cli_config()
        cli.init()

        mock_flyte.init_from_config.assert_called_once()
        call_cfg = mock_flyte.init_from_config.call_args[0][0]

        assert call_cfg.task.org is None

    @patch("flyte.cli._common.flyte")
    def test_cli_uses_config_platform_api_key_when_flyte_api_key_unset(self, mock_flyte, monkeypatch):
        """admin.apiKeyEnvVar resolution on PlatformConfig is used when FLYTE_API_KEY is not set."""
        monkeypatch.delenv("FLYTE_API_KEY", raising=False)
        cfg = Config(platform=PlatformConfig(api_key=API_KEY))
        cli = _make_cli_config(config=cfg)
        cli.init()

        mock_flyte.init_from_config.assert_called_once()
        call_cfg = mock_flyte.init_from_config.call_args[0][0]
        assert call_cfg.platform.client_id == "my-client-id"
        assert call_cfg.platform.client_credentials_secret == "my-secret"
        assert call_cfg.task.org == "my-org"
