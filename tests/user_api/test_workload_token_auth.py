"""Tests for workload identity (`WorkloadToken`) auth of task pods.

With workload identity on, the platform tells each task pod which action attempt it runs (`_U_WORKLOAD`) and mounts a
projected service-account token at `_U_WORKLOAD_TOKEN_FILE`. The pod presents that token, naming its attempt, instead of
the long-lived injected `EAGER_API_KEY`.
"""

from unittest.mock import AsyncMock, patch

import pytest

from flyte._initialize import init_in_cluster
from flyte.remote._client.auth._authenticators.factory import get_async_authenticator
from flyte.remote._client.auth._authenticators.workload_token import WorkloadTokenAuthenticator
from flyte.remote._client.auth.errors import AuthenticationError

WORKLOAD = "acme/p/development/r1/a0/1"


@pytest.fixture
def pod(monkeypatch, tmp_path):
    """A task pod: endpoint and api key injected, plus workload identity when enabled."""
    for var in ("UCTL_CONFIG", "FLYTECTL_CONFIG", "FLYTE_AUTH_TYPE", "FLYTE_ADMIN_AUTHTYPE", "FLYTE_AUTH_COMMAND"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("_U_EP_OVERRIDE", "dns:///acme.union.ai:443")
    monkeypatch.setenv("_UNION_EAGER_API_KEY", "legacy-composite-token")
    token = tmp_path / "token"
    token.write_text("projected-sa-token-1\n")
    monkeypatch.setenv("_U_WORKLOAD", WORKLOAD)
    monkeypatch.setenv("_U_WORKLOAD_TOKEN_FILE", str(token))
    return monkeypatch, token


class TestInitInCluster:
    @pytest.mark.asyncio
    async def test_workload_identity_replaces_the_api_key(self, pod):
        with patch("flyte._initialize.init", new_callable=AsyncMock) as mock_init:
            controller_kwargs = await init_in_cluster.aio()
        assert controller_kwargs["auth_type"] == "WorkloadToken"
        assert "api_key" not in controller_kwargs
        init_kwargs = mock_init.aio.await_args.kwargs
        assert init_kwargs["auth_type"] == "WorkloadToken"
        assert init_kwargs.get("api_key") is None

    @pytest.mark.asyncio
    async def test_explicit_auth_type_still_wins(self, pod):
        env, _ = pod
        env.setenv("FLYTE_AUTH_TYPE", "ExternalCommand")
        env.setenv("FLYTE_AUTH_COMMAND", "mint-token")
        with patch("flyte._initialize.init", new_callable=AsyncMock):
            controller_kwargs = await init_in_cluster.aio()
        assert controller_kwargs["auth_type"] == "ExternalCommand"

    @pytest.mark.asyncio
    async def test_without_a_token_file_the_api_key_is_used(self, pod):
        _, token = pod
        token.unlink()
        with patch("flyte._initialize.init", new_callable=AsyncMock):
            controller_kwargs = await init_in_cluster.aio()
        assert controller_kwargs["api_key"] == "legacy-composite-token"
        assert "auth_type" not in controller_kwargs

    @pytest.mark.asyncio
    async def test_explicit_api_key_wins(self, pod):
        with patch("flyte._initialize.init", new_callable=AsyncMock):
            controller_kwargs = await init_in_cluster.aio(api_key="explicit")
        assert controller_kwargs["api_key"] == "explicit"


class TestWorkloadTokenAuthenticator:
    @pytest.mark.asyncio
    async def test_sends_the_current_projected_token_and_attempt(self, pod):
        _, token = pod
        auth = WorkloadTokenAuthenticator(endpoint="dns:///acme.union.ai:443")
        headers = await auth.get_auth_headers()
        assert headers.headers["authorization"] == "Bearer projected-sa-token-1"
        assert headers.headers["x-union-workload"] == WORKLOAD

        # The kubelet rotates the projected token; it is read again once the last read is a minute old.
        token.write_text("projected-sa-token-2")
        with patch("flyte.remote._client.auth._authenticators.workload_token.time.monotonic") as mono:
            mono.return_value = auth._read_at + 30
            assert (await auth.get_auth_headers()).headers["authorization"] == "Bearer projected-sa-token-1"
            mono.return_value = auth._read_at + 61
            assert (await auth.get_auth_headers()).headers["authorization"] == "Bearer projected-sa-token-2"

        # A refused request reads it again at once.
        token.write_text("projected-sa-token-3")
        await auth.refresh_credentials()
        assert (await auth.get_auth_headers()).headers["authorization"] == "Bearer projected-sa-token-3"

    @pytest.mark.asyncio
    async def test_unreadable_or_empty_token_file(self, pod):
        _, token = pod
        auth = WorkloadTokenAuthenticator(endpoint="dns:///acme.union.ai:443")
        token.write_text("  \n")
        with pytest.raises(AuthenticationError, match="empty"):
            await auth.refresh_credentials()
        token.unlink()
        with pytest.raises(AuthenticationError, match="projected token"):
            await auth.refresh_credentials()

    def test_needs_the_workload_env(self, monkeypatch):
        monkeypatch.delenv("_U_WORKLOAD", raising=False)
        monkeypatch.delenv("_U_WORKLOAD_TOKEN_FILE", raising=False)
        with pytest.raises(AuthenticationError):
            WorkloadTokenAuthenticator(endpoint="dns:///acme.union.ai:443")

    def test_factory(self, pod):
        auth = get_async_authenticator("dns:///acme.union.ai:443", None, auth_type="WorkloadToken")
        assert isinstance(auth, WorkloadTokenAuthenticator)


class TestRustController:
    """`init_in_cluster`'s kwargs reach the Rust controller as a workload-token controller."""

    @pytest.fixture
    def fake_rust(self, monkeypatch):
        import sys
        import types
        from unittest.mock import MagicMock

        from flyte._internal.controllers import _ControllerState

        module = types.ModuleType("flyte._internal.controllers.remote._r_controller")
        module.RemoteController = MagicMock(name="RemoteController")
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setattr(_ControllerState, "controller", None)
        return module.RemoteController

    @pytest.mark.parametrize(
        "insecure,expected",
        [(False, "https://acme.union.ai:443"), (True, "http://acme.union.ai:443")],
    )
    def test_workload_token(self, fake_rust, insecure, expected):
        from flyte._internal.controllers import create_controller

        create_controller("rust", endpoint="dns:///acme.union.ai:443", insecure=insecure, auth_type="WorkloadToken")
        kwargs = fake_rust.call_args.kwargs
        assert kwargs["endpoint"] == expected
        assert kwargs["workload_token"] is True

    def test_without_workload_token(self, fake_rust):
        from flyte._internal.controllers import create_controller

        create_controller("rust", endpoint="localhost:8090", insecure=True)
        kwargs = fake_rust.call_args.kwargs
        assert kwargs["endpoint"] == "http://localhost:8090"
        assert kwargs["workload_token"] is False
