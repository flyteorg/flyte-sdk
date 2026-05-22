from pathlib import Path

from flyte.config._config import PlatformConfig


def test_platform_config_reads_auth_override_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """admin:
  endpoint: dns:///dogfood.cloud-staging.union.ai
  clientId: dogfood-flyteadmin
  scopes:
    - all
  authorizationHeader: flyte-authorization
  redirectURL: http://localhost:53593/callback
  audience: dogfood-audience
"""
    )

    cfg = PlatformConfig.auto(config_path)

    assert cfg.endpoint == "dns:///dogfood.cloud-staging.union.ai"
    assert cfg.client_id == "dogfood-flyteadmin"
    assert cfg.scopes == ["all"]
    assert cfg.authorization_header == "flyte-authorization"
    assert cfg.redirect_uri == "http://localhost:53593/callback"
    assert cfg.audience == "dogfood-audience"
