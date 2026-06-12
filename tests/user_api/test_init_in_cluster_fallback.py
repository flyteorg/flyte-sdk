"""Tests for init_in_cluster's file-mounted config fallback.

When the chart-managed file-mount deploy sets FLYTECTL_CONFIG (or
UCTL_CONFIG) on the task pod, init_in_cluster picks the path up
directly and delegates to init_from_config instead of reading the
legacy _UNION_EAGER_API_KEY env var. resolve_config_path's full
precedence chain is deliberately NOT walked here — it does a
`git rev-parse` subprocess that is wasted work in a task pod.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from flyte._initialize import _platform_to_client_kwargs, init_in_cluster
from flyte.config import Config
from flyte.config._config import PlatformConfig
from flyte.errors import InitializationError


@pytest.fixture
def yaml_config(tmp_path: Path) -> Path:
    """Write a minimal valid SDK config (and the secret it references) to tmp."""
    secret = tmp_path / "client_secret"
    secret.write_text("test-secret-value")
    p = tmp_path / "config.yaml"
    p.write_text(
        f"""admin:
  endpoint: dns:///example.com:443
  clientId: test-client
  clientSecretLocation: {secret}
  authType: ClientSecret
"""
    )
    return p


# init_from_config is @syncify-decorated; init_in_cluster calls its
# ``.aio`` form to stay inside the async context, so the tests patch
# the ``.aio`` attribute as an AsyncMock.
INIT_FROM_CONFIG_AIO = "flyte._initialize.init_from_config.aio"


class TestInitInClusterFallback:
    @pytest.mark.asyncio
    async def test_flytectl_config_env_var_is_picked_up(self, yaml_config, monkeypatch):
        """FLYTECTL_CONFIG pointing at a real file -> init_in_cluster
        delegates to init_from_config and returns controller kwargs
        derived from the same PlatformConfig.
        """
        monkeypatch.delenv("_UNION_EAGER_API_KEY", raising=False)
        monkeypatch.delenv("EAGER_API_KEY", raising=False)
        monkeypatch.delenv("UCTL_CONFIG", raising=False)
        monkeypatch.setenv("FLYTECTL_CONFIG", str(yaml_config))

        with patch(INIT_FROM_CONFIG_AIO, new_callable=AsyncMock) as mock_init_from_config:
            result = await init_in_cluster.aio()

        mock_init_from_config.assert_awaited_once()
        assert mock_init_from_config.call_args.kwargs["path_or_config"] == str(yaml_config)
        # Returned kwargs match create_remote_controller's signature so the
        # runtime caller can spread them directly.
        assert result["endpoint"] == "dns:///example.com:443"
        assert result["client_id"] == "test-client"
        assert result["client_credentials_secret"] == "test-secret-value"
        assert result["auth_type"] == "ClientSecret"
        assert result["headless"] is True

    @pytest.mark.asyncio
    async def test_org_propagates_from_env_to_init_from_config(self, yaml_config, monkeypatch):
        """Runtime org (via _U_ORG_NAME) must reach init_from_config so the
        config-file fallback doesn't silently lose it. The legacy api-key
        branch threads org=os.getenv("_U_ORG_NAME") into init.aio(org=...);
        the fallback path must preserve the same semantics.
        """
        monkeypatch.delenv("_UNION_EAGER_API_KEY", raising=False)
        monkeypatch.delenv("EAGER_API_KEY", raising=False)
        monkeypatch.delenv("UCTL_CONFIG", raising=False)
        monkeypatch.setenv("FLYTECTL_CONFIG", str(yaml_config))
        monkeypatch.setenv("_U_ORG_NAME", "apple")

        with patch(INIT_FROM_CONFIG_AIO, new_callable=AsyncMock) as mock_init_from_config:
            await init_in_cluster.aio()

        mock_init_from_config.assert_awaited_once()
        assert mock_init_from_config.call_args.kwargs.get("org") == "apple"

    @pytest.mark.asyncio
    async def test_explicit_org_kwarg_wins_over_env_in_fallback(self, yaml_config, monkeypatch):
        """An org explicitly passed to init_in_cluster takes precedence over
        the _U_ORG_NAME env var, same shape as the legacy branch."""
        monkeypatch.delenv("_UNION_EAGER_API_KEY", raising=False)
        monkeypatch.delenv("EAGER_API_KEY", raising=False)
        monkeypatch.delenv("UCTL_CONFIG", raising=False)
        monkeypatch.setenv("FLYTECTL_CONFIG", str(yaml_config))
        monkeypatch.setenv("_U_ORG_NAME", "ignored")

        with patch(INIT_FROM_CONFIG_AIO, new_callable=AsyncMock) as mock_init_from_config:
            await init_in_cluster.aio(org="explicit")

        mock_init_from_config.assert_awaited_once()
        assert mock_init_from_config.call_args.kwargs.get("org") == "explicit"

    @pytest.mark.asyncio
    async def test_uctl_config_env_var_takes_precedence_over_flytectl(self, yaml_config, monkeypatch):
        """UCTL_CONFIG matches uctl/flytectl convention: UCTL_CONFIG wins
        when both are set, since the Union CLI is the newer surface.
        """
        monkeypatch.delenv("_UNION_EAGER_API_KEY", raising=False)
        monkeypatch.delenv("EAGER_API_KEY", raising=False)
        monkeypatch.setenv("UCTL_CONFIG", str(yaml_config))
        monkeypatch.setenv("FLYTECTL_CONFIG", "/non/existent/path.yaml")

        with patch(INIT_FROM_CONFIG_AIO, new_callable=AsyncMock) as mock_init_from_config:
            await init_in_cluster.aio()

        mock_init_from_config.assert_awaited_once()
        assert mock_init_from_config.call_args.kwargs["path_or_config"] == str(yaml_config)

    @pytest.mark.asyncio
    async def test_no_fallback_when_env_var_unset(self, monkeypatch):
        """Neither env var set -> legacy api-key/env-var branch runs."""
        monkeypatch.delenv("UCTL_CONFIG", raising=False)
        monkeypatch.delenv("FLYTECTL_CONFIG", raising=False)
        monkeypatch.setenv("_UNION_EAGER_API_KEY", "legacy-composite-token")

        with patch(INIT_FROM_CONFIG_AIO, new_callable=AsyncMock) as mock_init_from_config:
            with patch("flyte._initialize.init", new_callable=AsyncMock):
                await init_in_cluster.aio()

        mock_init_from_config.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_path_surfaces_loud_error(self, monkeypatch, tmp_path):
        """Env var set to a path that does not exist -> init_from_config
        raises InitializationError("ConfigFileNotFoundError"). A typo in
        FLYTECTL_CONFIG must NOT silently route to the legacy api-key
        branch — that would mask the misconfiguration. The chart
        contract is "env var points at a real mounted file"; anything
        else is loud."""
        monkeypatch.delenv("UCTL_CONFIG", raising=False)
        missing = tmp_path / "missing.yaml"
        monkeypatch.setenv("FLYTECTL_CONFIG", str(missing))
        monkeypatch.setenv("_UNION_EAGER_API_KEY", "legacy-composite-token")

        with pytest.raises(InitializationError, match="does not exist"):
            await init_in_cluster.aio()

    @pytest.mark.asyncio
    async def test_no_fallback_when_explicit_api_key_passed(self, yaml_config, monkeypatch):
        """Explicit api_key kwarg wins over file-mount fallback even
        when FLYTECTL_CONFIG is set."""
        monkeypatch.setenv("FLYTECTL_CONFIG", str(yaml_config))

        with patch(INIT_FROM_CONFIG_AIO, new_callable=AsyncMock) as mock_init_from_config:
            with patch("flyte._initialize.init", new_callable=AsyncMock):
                await init_in_cluster.aio(api_key="explicit-key")

        mock_init_from_config.assert_not_awaited()

    def test_platform_kwargs_helper_covers_every_used_field(self, yaml_config):
        """Drift catcher: the helper is the single source of truth shared
        by init_from_config (laptop/SDK Client path) and init_in_cluster
        (in-pod runtime-controller path). If a new PlatformConfig field
        needs to reach the HTTP/auth stack, both paths must pick it up;
        this test asserts that adding it to the helper is sufficient.
        """
        cfg = Config.auto(str(yaml_config))
        kw = _platform_to_client_kwargs(cfg.platform)
        # Fields the chart-mounted in-cluster config sets — these MUST
        # round-trip through the helper or the controller misconfigures.
        assert kw["endpoint"] == cfg.platform.endpoint
        assert kw["client_id"] == cfg.platform.client_id
        assert kw["client_credentials_secret"] == cfg.platform.client_credentials_secret
        assert kw["auth_type"] == cfg.platform.auth_mode

    def test_platform_kwargs_helper_omits_disable_keyring(self):
        """disable_keyring is intentionally NOT in the helper output:
        the dict is spread into create_remote_controller from
        init_in_cluster, and the controller's constructor does not
        accept disable_keyring. Threading it would raise TypeError on
        every in-cluster task pod. init_from_config passes
        disable_keyring to init separately, alongside the helper."""
        pc = PlatformConfig(
            endpoint="dns:///example.com:443",
            disable_keyring=True,
        )
        kw = _platform_to_client_kwargs(pc)
        assert "disable_keyring" not in kw

    def test_platform_kwargs_helper_prefers_ca_file_over_skip_verify(self):
        """When both ca_cert_file_path and insecure_skip_verify are set,
        the helper picks ca_cert_file_path. _resolve_tls_ca_cert routes
        on the first present flag and the bootstrap-from-server path
        only fetches the leaf, which trips UnknownIssuer when nginx is
        configured to serve the leaf alone (no intermediates)."""
        pc = PlatformConfig(
            endpoint="dns:///example.com:443",
            ca_cert_file_path="/etc/flyte/credentials/ca.crt",
            insecure_skip_verify=True,
        )
        kw = _platform_to_client_kwargs(pc)
        assert kw["ca_cert_file_path"] == "/etc/flyte/credentials/ca.crt"
        assert "insecure_skip_verify" not in kw
