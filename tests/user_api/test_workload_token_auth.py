"""Tests for the dataplane STS (`WorkloadToken`) auth of task pods.

When the platform runs the dataplane STS it points each task pod at it with
`_U_STS_ENDPOINT` and mounts a projected service-account token at
`_U_STS_TOKEN_FILE`. The pod then exchanges that token for a short-lived one
instead of using the long-lived injected `EAGER_API_KEY`.
"""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from flyte._initialize import init_in_cluster
from flyte.remote._client.auth._authenticators.factory import get_async_authenticator
from flyte.remote._client.auth._authenticators.workload_token import WorkloadTokenAuthenticator
from flyte.remote._client.auth.errors import AuthenticationError

STS = "http://union-operator-sts.union.svc:8095/oauth2/token"


@pytest.fixture
def pod(monkeypatch, tmp_path):
    """A task pod: endpoint and api key injected, plus the STS when enabled."""
    for var in ("UCTL_CONFIG", "FLYTECTL_CONFIG", "FLYTE_AUTH_TYPE", "FLYTE_ADMIN_AUTHTYPE", "FLYTE_AUTH_COMMAND"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("_U_EP_OVERRIDE", "dns:///acme.union.ai:443")
    monkeypatch.setenv("_UNION_EAGER_API_KEY", "legacy-composite-token")
    token = tmp_path / "token"
    token.write_text("projected-sa-token-1\n")
    monkeypatch.setenv("_U_STS_ENDPOINT", STS)
    monkeypatch.setenv("_U_STS_TOKEN_FILE", str(token))
    return monkeypatch, token


class TestInitInCluster:
    @pytest.mark.asyncio
    async def test_sts_replaces_the_api_key(self, pod):
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


def sts_session(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestWorkloadTokenAuthenticator:
    @pytest.mark.asyncio
    async def test_exchanges_the_current_projected_token(self, pod):
        _, token = pod
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == STS
            form = dict(httpx.QueryParams(request.content.decode()))
            seen.append(form)
            return httpx.Response(
                200, json={"access_token": f"sts-{len(seen)}", "token_type": "Bearer", "expires_in": 600}
            )

        auth = WorkloadTokenAuthenticator(endpoint="dns:///acme.union.ai:443", sts_http_session=sts_session(handler))
        await auth.refresh_credentials()
        assert auth.get_credentials().access_token == "sts-1"
        assert seen[0] == {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
            "subject_token": "projected-sa-token-1",
        }
        headers = await auth.get_auth_headers()
        assert headers.headers["authorization"] == "Bearer sts-1"

        # The kubelet rotates the projected token; each refresh reads it again.
        token.write_text("projected-sa-token-2")
        await auth.refresh_credentials()
        assert seen[1]["subject_token"] == "projected-sa-token-2"

    @pytest.mark.asyncio
    async def test_refreshes_before_expiry(self, pod):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(
                200, json={"access_token": f"t{len(calls)}", "token_type": "Bearer", "expires_in": 10}
            )

        auth = WorkloadTokenAuthenticator(endpoint="dns:///acme.union.ai:443", sts_http_session=sts_session(handler))
        await auth.refresh_credentials()
        with patch("flyte.remote._client.auth._authenticators.workload_token.time.monotonic") as mono:
            mono.return_value = auth._obtained_at + 5
            assert (await auth.get_auth_headers()).headers["authorization"] == "Bearer t1"
            mono.return_value = auth._obtained_at + 9
            assert (await auth.get_auth_headers()).headers["authorization"] == "Bearer t2"
        assert len(calls) == 2

    @pytest.mark.parametrize(
        "status,body",
        [
            (400, {"error": "invalid_grant"}),
            (503, {"error": "temporarily_unavailable"}),
            (200, {"access_token": "x", "token_type": "mac"}),
            (200, {"token_type": "Bearer"}),
        ],
    )
    @pytest.mark.asyncio
    async def test_refusals_are_auth_errors(self, pod, status, body):
        auth = WorkloadTokenAuthenticator(
            endpoint="dns:///acme.union.ai:443",
            sts_http_session=sts_session(lambda r: httpx.Response(status, content=json.dumps(body))),
        )
        with pytest.raises(AuthenticationError):
            await auth.refresh_credentials()

    @pytest.mark.asyncio
    async def test_unreadable_token_file(self, pod):
        _, token = pod
        token.unlink()
        auth = WorkloadTokenAuthenticator(
            endpoint="dns:///acme.union.ai:443", sts_http_session=sts_session(lambda r: httpx.Response(500))
        )
        with pytest.raises(AuthenticationError, match="projected token"):
            await auth.refresh_credentials()

    def test_needs_the_sts_env(self, monkeypatch):
        monkeypatch.delenv("_U_STS_ENDPOINT", raising=False)
        monkeypatch.delenv("_U_STS_TOKEN_FILE", raising=False)
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
