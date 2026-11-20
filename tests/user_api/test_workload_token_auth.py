"""Tests for workload identity (`WorkloadToken`) auth of task pods.

With workload identity on, the platform tells each task pod which action attempt it runs (`_U_WORKLOAD`) and mounts a
projected service-account token at `_U_WORKLOAD_TOKEN_FILE`. The pod presents that token, naming its attempt, instead of
the long-lived injected `EAGER_API_KEY`.
"""

from unittest.mock import AsyncMock, patch

import pytest
from connectrpc.code import Code

from flyte._initialize import init_in_cluster
from flyte.remote._client.auth._authenticators.factory import get_async_authenticator
from flyte.remote._client.auth._authenticators.workload_token import WorkloadTokenAuthenticator
from flyte.remote._client.auth.errors import AuthenticationError

WORKLOAD = "acme/p/development/r1/a0/1"


@pytest.fixture
def pod(monkeypatch, tmp_path):
    """A task pod: endpoint and api key injected, plus workload identity when enabled."""
    for var in (
        "UCTL_CONFIG",
        "FLYTECTL_CONFIG",
        "FLYTE_AUTH_TYPE",
        "FLYTE_ADMIN_AUTHTYPE",
        "FLYTE_AUTH_COMMAND",
        "FLYTE_AUTH_PREFER_WORKLOAD_IDENTITY",
    ):
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
    async def test_workload_identity_wins_over_the_auth_type_when_preferred(self, pod):
        """A cluster that sets one auth type on every pod opts pods with a workload identity into it."""
        env, _ = pod
        env.setenv("FLYTE_AUTH_TYPE", "ExternalCommand")
        env.setenv("FLYTE_AUTH_COMMAND", "mint-token")
        env.setenv("FLYTE_AUTH_PREFER_WORKLOAD_IDENTITY", "1")
        with patch("flyte._initialize.init", new_callable=AsyncMock):
            controller_kwargs = await init_in_cluster.aio()
        assert controller_kwargs["auth_type"] == "WorkloadToken"
        assert "command" not in controller_kwargs
        assert "api_key" not in controller_kwargs

    @pytest.mark.asyncio
    async def test_preferring_workload_identity_keeps_the_auth_type_without_one(self, pod):
        """On a pod the platform gave no workload identity, the preference changes nothing."""
        env, token = pod
        token.unlink()
        env.setenv("FLYTE_AUTH_TYPE", "ExternalCommand")
        env.setenv("FLYTE_AUTH_COMMAND", "mint-token")
        env.setenv("FLYTE_AUTH_PREFER_WORKLOAD_IDENTITY", "1")
        with patch("flyte._initialize.init", new_callable=AsyncMock):
            controller_kwargs = await init_in_cluster.aio()
        assert controller_kwargs["auth_type"] == "ExternalCommand"
        assert controller_kwargs["command"] == ["mint-token"]
        assert "api_key" not in controller_kwargs

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

    @staticmethod
    def _fallback_stub():
        from flyte.remote._client.auth._authenticators.base import AuthHeaders

        stub = AsyncMock()
        stub.get_auth_headers.return_value = AuthHeaders(creds_id="key", headers={"authorization": "Bearer api-key"})
        built = []

        def factory():
            built.append(stub)
            return stub

        return factory, stub, built

    @pytest.mark.asyncio
    async def test_falls_back_to_the_mounted_api_key_when_the_token_is_refused(self, pod):
        """The token was re-read and is unchanged, so it was refused for itself, not for being stale."""
        factory, stub, built = self._fallback_stub()
        auth = WorkloadTokenAuthenticator(endpoint="dns:///acme.union.ai:443", fallback=factory)
        first = await auth.get_auth_headers()
        assert first.headers["authorization"] == "Bearer projected-sa-token-1"

        auth.note_refusal(Code.UNAUTHENTICATED, first.creds_id)  # the interceptor, after a refusal
        await auth.refresh_credentials(first.creds_id)

        assert len(built) == 1
        stub.refresh_credentials.assert_awaited_once()
        headers = await auth.get_auth_headers()
        assert headers.headers == {"authorization": "Bearer api-key"}
        assert "x-union-workload" not in headers.headers
        await auth.refresh_credentials("key")  # later refusals refresh the key, never switch back
        assert len(built) == 1

    @pytest.mark.asyncio
    async def test_a_rotated_token_is_tried_before_falling_back(self, pod):
        _, token = pod
        factory, _, built = self._fallback_stub()
        auth = WorkloadTokenAuthenticator(endpoint="dns:///acme.union.ai:443", fallback=factory)
        first = await auth.get_auth_headers()
        token.write_text("projected-sa-token-2")

        auth.note_refusal(Code.UNAUTHENTICATED, first.creds_id)
        await auth.refresh_credentials(first.creds_id)

        assert built == []
        assert (await auth.get_auth_headers()).headers["authorization"] == "Bearer projected-sa-token-2"

    @pytest.mark.asyncio
    async def test_scheduled_rereads_never_fall_back(self, pod):
        factory, _, built = self._fallback_stub()
        auth = WorkloadTokenAuthenticator(endpoint="dns:///acme.union.ai:443", fallback=factory)
        await auth.get_auth_headers()
        with patch("flyte.remote._client.auth._authenticators.workload_token.time.monotonic") as mono:
            mono.return_value = auth._read_at + 61
            await auth.get_auth_headers()
        assert built == []

    @pytest.mark.asyncio
    async def test_without_a_mounted_key_a_refusal_stays_on_the_token(self, pod):
        auth = WorkloadTokenAuthenticator(endpoint="dns:///acme.union.ai:443")
        first = await auth.get_auth_headers()
        auth.note_refusal(Code.UNAUTHENTICATED, first.creds_id)
        await auth.refresh_credentials(first.creds_id)
        assert (await auth.get_auth_headers()).headers["authorization"] == "Bearer projected-sa-token-1"

    @pytest.mark.asyncio
    async def test_other_retriable_errors_never_fall_back(self, pod):
        """UNKNOWN, or a message that merely mentions authorization, only re-reads the token."""
        factory, _, built = self._fallback_stub()
        auth = WorkloadTokenAuthenticator(endpoint="dns:///acme.union.ai:443", fallback=factory)
        first = await auth.get_auth_headers()
        auth.note_refusal(Code.UNKNOWN, first.creds_id)
        await auth.refresh_credentials(first.creds_id)
        assert built == []
        assert "x-union-workload" in (await auth.get_auth_headers()).headers

    @pytest.mark.asyncio
    async def test_a_refusal_of_an_older_token_never_falls_back(self, pod):
        """Another request already moved to a rotated token: this refusal says nothing about the current one."""
        _, token = pod
        factory, _, built = self._fallback_stub()
        auth = WorkloadTokenAuthenticator(endpoint="dns:///acme.union.ai:443", fallback=factory)
        old = await auth.get_auth_headers()
        token.write_text("projected-sa-token-2")
        await auth.refresh_credentials()  # a concurrent request's refresh moved to the rotated token
        auth.note_refusal(Code.UNAUTHENTICATED, old.creds_id)
        await auth.refresh_credentials(old.creds_id)
        assert built == []
        assert (await auth.get_auth_headers()).headers["authorization"] == "Bearer projected-sa-token-2"

    @pytest.mark.asyncio
    async def test_interceptor_reports_the_refusal(self, pod):
        """The unary interceptor tells the authenticator what was refused before it refreshes."""
        from connectrpc.errors import ConnectError

        from flyte.remote._client.auth._interceptors.auth import AuthUnaryInterceptor

        factory, _, built = self._fallback_stub()
        auth = WorkloadTokenAuthenticator(endpoint="dns:///acme.union.ai:443", fallback=factory)
        interceptor = AuthUnaryInterceptor(lambda: auth)

        class Ctx:
            def __init__(self):
                self._h = {}

            def request_headers(self):
                return self._h

        sent = []

        async def call_next(request, ctx):
            sent.append(dict(ctx.request_headers()))
            if len(sent) == 1:
                raise ConnectError(Code.UNAUTHENTICATED, "workload token rejected")
            return "ok"

        assert await interceptor.intercept_unary(call_next, object(), Ctx()) == "ok"
        assert sent[0]["authorization"] == "Bearer projected-sa-token-1"
        assert sent[1]["authorization"] == "Bearer api-key"
        assert len(built) == 1

    def test_fallback_is_built_only_from_a_mounted_key(self, monkeypatch):
        from flyte.remote._client.auth._authenticators.workload_token import mounted_api_key_fallback

        for var in ("_UNION_EAGER_API_KEY", "EAGER_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        assert mounted_api_key_fallback("dns:///acme.union.ai:443", None) is None
        monkeypatch.setenv("_UNION_EAGER_API_KEY", "anything")
        assert callable(mounted_api_key_fallback("dns:///acme.union.ai:443", None))

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

    def test_workload_token_bare_endpoint_keeps_plaintext(self, fake_rust):
        """A bare host:port is a local, plaintext endpoint, as it was before workload tokens."""
        from flyte._internal.controllers import create_controller

        create_controller("rust", endpoint="flyte.flyte.svc:8090", auth_type="WorkloadToken")
        kwargs = fake_rust.call_args.kwargs
        assert kwargs["endpoint"] == "http://flyte.flyte.svc:8090"
        assert kwargs["workload_token"] is True

    def test_without_workload_token(self, fake_rust):
        from flyte._internal.controllers import create_controller

        create_controller("rust", endpoint="localhost:8090", insecure=True)
        kwargs = fake_rust.call_args.kwargs
        assert kwargs["endpoint"] == "http://localhost:8090"
        assert kwargs["workload_token"] is False


def test_rusty_controller_takes_the_platform_endpoint_without_an_api_key(monkeypatch, tmp_path):
    """A reusable worker (rusty) with no api key, as under workload identity, connects to the
    endpoint the platform injected (`_U_EP_OVERRIDE`), not a hard-coded default."""
    import asyncio

    from flyte._internal.runtime import rusty

    token = tmp_path / "token"
    token.write_text("t")
    monkeypatch.setenv("_U_WORKLOAD", "pod/acme/p/development/c1/ns/env-0")
    monkeypatch.setenv("_U_WORKLOAD_TOKEN_FILE", str(token))
    monkeypatch.setenv("_U_EP_OVERRIDE", "dns:///acme.cloud.union.ai")
    monkeypatch.delenv("EAGER_API_KEY", raising=False)
    monkeypatch.delenv("_UNION_EAGER_API_KEY", raising=False)
    seen = {}
    monkeypatch.setattr(rusty, "_create_controller", lambda ct, **kw: seen.update(kw) or object())
    asyncio.run(rusty.create_controller())
    assert seen["endpoint"] == "dns:///acme.cloud.union.ai"
    assert seen.get("auth_type") == "WorkloadToken"


def test_rusty_platform_endpoint_wins_over_the_executor_default(monkeypatch, tmp_path):
    """The reusable-worker executor passes `host.docker.internal:8090` (insecure) whenever it has no
    api key; in a task pod the platform's `_U_EP_OVERRIDE` wins, over TLS."""
    import asyncio

    from flyte._internal.runtime import rusty

    token = tmp_path / "token"
    token.write_text("t")
    monkeypatch.setenv("_U_WORKLOAD", "pod/acme/p/development/c1/ns/env-0")
    monkeypatch.setenv("_U_WORKLOAD_TOKEN_FILE", str(token))
    monkeypatch.setenv("_U_EP_OVERRIDE", "dns:///acme.cloud.union.ai")
    monkeypatch.delenv("_U_INSECURE", raising=False)
    monkeypatch.delenv("EAGER_API_KEY", raising=False)
    monkeypatch.delenv("_UNION_EAGER_API_KEY", raising=False)
    seen = {}
    monkeypatch.setattr(rusty, "_create_controller", lambda ct, **kw: seen.update(kw) or object())
    asyncio.run(rusty.create_controller(endpoint="dns:///host.docker.internal:8090", insecure=True, api_key=None))
    assert seen["endpoint"] == "dns:///acme.cloud.union.ai"
    assert seen["insecure"] is False
    assert seen.get("auth_type") == "WorkloadToken"


def test_rusty_keeps_the_callers_endpoint_off_platform(monkeypatch):
    """Locally (no `_U_EP_OVERRIDE`) the caller's endpoint and insecure flag still apply."""
    import asyncio

    from flyte._internal.runtime import rusty

    for k in ("_U_EP_OVERRIDE", "_U_WORKLOAD", "_U_WORKLOAD_TOKEN_FILE", "EAGER_API_KEY", "_UNION_EAGER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    seen = {}
    monkeypatch.setattr(rusty, "_create_controller", lambda ct, **kw: seen.update(kw) or object())
    asyncio.run(rusty.create_controller(endpoint="dns:///host.docker.internal:8090", insecure=True))
    assert seen["endpoint"] == "dns:///host.docker.internal:8090"
    assert seen["insecure"] is True
