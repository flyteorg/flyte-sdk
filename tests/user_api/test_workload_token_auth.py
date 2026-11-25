"""Tests for workload identity (`WorkloadToken`) auth of task pods.

With workload identity on, the platform tells each task pod which action attempt it runs (`_U_WORKLOAD`) and mounts a
projected service-account token at `_U_WORKLOAD_TOKEN_FILE`. The pod presents that token, naming its attempt, instead of
the long-lived injected `EAGER_API_KEY`. It sends one bearer credential, shaped as an unsigned compact JWS, that carries
the token, the workload and (for a pod workload) the action a request is made for.
"""

import base64
import json
from unittest.mock import AsyncMock, patch

import pytest
from connectrpc.code import Code

from flyte._initialize import init_in_cluster
from flyte.remote._client.auth._authenticators.factory import get_async_authenticator
from flyte.remote._client.auth._authenticators.workload_token import WorkloadTokenAuthenticator
from flyte.remote._client.auth.errors import AuthenticationError

WORKLOAD = "acme/p/development/r1/a0/1"
POD_WORKLOAD = "pod/acme/p/development/c1/ns/worker-0"


def _b64decode(segment: str) -> bytes:
    assert "=" not in segment, "segments are base64url without padding"
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


def decode_credential(headers: dict) -> dict:
    """The payload of the bearer credential in these headers, after checking its JWS shape."""
    assert not any(k.startswith("x-union-workload") for k in headers), headers
    scheme, credential = headers["authorization"].split(" ", 1)
    assert scheme == "Bearer"
    header, payload, signature = credential.split(".")
    assert signature == ""
    assert _b64decode(header) == b'{"typ":"union-workload","alg":"none"}'
    raw = _b64decode(payload)
    decoded = json.loads(raw.decode("utf-8"))
    assert raw == json.dumps(decoded, separators=(",", ":"), ensure_ascii=False).encode("utf-8"), "compact JSON"
    return decoded


def sent_token(headers: dict) -> str:
    return decode_credential(headers)["token"]


@pytest.fixture
def pod(monkeypatch, tmp_path):
    """A task pod: endpoint and api key injected, plus workload identity when enabled."""
    for var in (
        "UCTL_CONFIG",
        "FLYTECTL_CONFIG",
        "FLYTE_AUTH_TYPE",
        "FLYTE_ADMIN_AUTHTYPE",
        "FLYTE_AUTH_COMMAND",
        "FLYTE_AUTH_PRECEDENCE",
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
    async def test_precedence_puts_workload_identity_ahead_of_the_auth_type(self, pod):
        """A cluster that sets one auth type on every pod opts pods with a workload identity into it."""
        env, _ = pod
        env.setenv("FLYTE_AUTH_TYPE", "ExternalCommand")
        env.setenv("FLYTE_AUTH_COMMAND", "mint-token")
        env.setenv("FLYTE_AUTH_PRECEDENCE", "workload-identity")
        with patch("flyte._initialize.init", new_callable=AsyncMock):
            controller_kwargs = await init_in_cluster.aio()
        assert controller_kwargs["auth_type"] == "WorkloadToken"
        assert "command" not in controller_kwargs
        assert "api_key" not in controller_kwargs

    @pytest.mark.asyncio
    async def test_precedence_skips_a_source_the_pod_does_not_have(self, pod):
        """On a pod the platform gave no workload identity, preferring it changes nothing."""
        env, token = pod
        token.unlink()
        env.setenv("FLYTE_AUTH_TYPE", "ExternalCommand")
        env.setenv("FLYTE_AUTH_COMMAND", "mint-token")
        env.setenv("FLYTE_AUTH_PRECEDENCE", "workload-identity")
        with patch("flyte._initialize.init", new_callable=AsyncMock):
            controller_kwargs = await init_in_cluster.aio()
        assert controller_kwargs["auth_type"] == "ExternalCommand"
        assert controller_kwargs["command"] == ["mint-token"]
        assert "api_key" not in controller_kwargs

    @pytest.mark.asyncio
    async def test_precedence_can_put_the_api_key_first(self, pod):
        env, _ = pod
        env.setenv("FLYTE_AUTH_TYPE", "ExternalCommand")
        env.setenv("FLYTE_AUTH_COMMAND", "mint-token")
        env.setenv("FLYTE_AUTH_PRECEDENCE", "api-key")
        with patch("flyte._initialize.init", new_callable=AsyncMock):
            controller_kwargs = await init_in_cluster.aio()
        assert controller_kwargs["api_key"] == "legacy-composite-token"
        assert "auth_type" not in controller_kwargs

    @pytest.mark.asyncio
    async def test_precedence_ignores_names_it_does_not_know(self, pod):
        """A value written for a newer SDK still works: unknown sources are skipped, known ones honored."""
        env, _ = pod
        env.setenv("FLYTE_AUTH_TYPE", "ExternalCommand")
        env.setenv("FLYTE_AUTH_COMMAND", "mint-token")
        env.setenv("FLYTE_AUTH_PRECEDENCE", "some-future-source, Workload-Identity ,")
        with patch("flyte._initialize.init", new_callable=AsyncMock):
            controller_kwargs = await init_in_cluster.aio()
        assert controller_kwargs["auth_type"] == "WorkloadToken"

    def test_precedence_keeps_unlisted_sources_in_their_default_order(self, monkeypatch):
        from flyte._initialize import _auth_precedence

        monkeypatch.delenv("FLYTE_AUTH_PRECEDENCE", raising=False)
        assert _auth_precedence() == ["auth-env", "workload-identity", "api-key"]
        monkeypatch.setenv("FLYTE_AUTH_PRECEDENCE", "api-key")
        assert _auth_precedence() == ["api-key", "auth-env", "workload-identity"]
        monkeypatch.setenv("FLYTE_AUTH_PRECEDENCE", "api-key,workload-identity,api-key")
        assert _auth_precedence() == ["api-key", "workload-identity", "auth-env"]

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
        assert set(headers.headers) == {"authorization"}
        assert decode_credential(headers.headers) == {"token": "projected-sa-token-1", "workload": WORKLOAD}

        # The kubelet rotates the projected token; it is read again once the last read is a minute old.
        token.write_text("projected-sa-token-2")
        with patch("flyte.remote._client.auth._authenticators.workload_token.time.monotonic") as mono:
            mono.return_value = auth._read_at + 30
            assert sent_token((await auth.get_auth_headers()).headers) == "projected-sa-token-1"
            mono.return_value = auth._read_at + 61
            assert sent_token((await auth.get_auth_headers()).headers) == "projected-sa-token-2"

        # A refused request reads it again at once.
        token.write_text("projected-sa-token-3")
        await auth.refresh_credentials()
        assert sent_token((await auth.get_auth_headers()).headers) == "projected-sa-token-3"

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
        assert sent_token(first.headers) == "projected-sa-token-1"

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
        assert sent_token((await auth.get_auth_headers()).headers) == "projected-sa-token-2"

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
        assert sent_token((await auth.get_auth_headers()).headers) == "projected-sa-token-1"

    @pytest.mark.asyncio
    async def test_other_retriable_errors_never_fall_back(self, pod):
        """UNKNOWN, or a message that merely mentions authorization, only re-reads the token."""
        factory, _, built = self._fallback_stub()
        auth = WorkloadTokenAuthenticator(endpoint="dns:///acme.union.ai:443", fallback=factory)
        first = await auth.get_auth_headers()
        auth.note_refusal(Code.UNKNOWN, first.creds_id)
        await auth.refresh_credentials(first.creds_id)
        assert built == []
        assert decode_credential((await auth.get_auth_headers()).headers)["workload"] == WORKLOAD

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
        assert sent_token((await auth.get_auth_headers()).headers) == "projected-sa-token-2"

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
        assert sent_token(sent[0]) == "projected-sa-token-1"
        assert sent[1] == {"authorization": "Bearer api-key"}
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


class _Ctx:
    """A connectrpc RequestContext with the headers the call site passed."""

    def __init__(self, headers=None):
        self._h = dict(headers or {})

    def request_headers(self):
        return self._h


async def _send(auth, headers=None) -> dict:
    """Sends a unary request through the auth interceptor and returns the headers it went out with."""
    from flyte.remote._client.auth._interceptors.auth import AuthUnaryInterceptor

    sent = []

    async def call_next(request, ctx):
        sent.append(dict(ctx.request_headers()))
        return "ok"

    await AuthUnaryInterceptor(lambda: auth).intercept_unary(call_next, object(), _Ctx(headers))
    return sent[0]


def _task_context(run_name: str, action_name: str):
    from flyte.models import ActionID, RawDataPath, TaskContext

    return TaskContext(
        action=ActionID(name=action_name, run_name=run_name, project="p", domain="development", org="acme"),
        run_base_dir="s3://bucket/run",
        output_path="s3://bucket/run/out",
        raw_data_path=RawDataPath(path="s3://bucket/raw"),
        version="v1",
        report=None,
    )


def _launch_controller(auth, *, actions_service: bool):
    """A controller whose enqueue goes through the auth interceptor, recording the headers sent."""
    import asyncio

    from aiolimiter import AsyncLimiter

    from flyte._internal.controllers.remote._core import Controller

    controller = object.__new__(Controller)
    controller._running = True
    controller._shared_queue = asyncio.Queue()
    controller._rate_limiter = AsyncLimiter(100, 1.0)
    controller._enqueue_timeout = 1.0
    controller._consecutive_launch_timeouts = 0
    sent = []

    async def enqueue(_req, timeout_ms=None, headers=None):
        sent.append(await _send(auth, headers))

    service = AsyncMock()
    if actions_service:
        service.enqueue.side_effect = enqueue
        controller._actions_service = service
        controller._queue_service = None
    else:
        service.enqueue_action.side_effect = enqueue
        controller._actions_service = None
        controller._queue_service = service
    return controller, sent


def _child(run_name: str = "r1", parent: str = "a0"):
    from flyteidl2.common import identifier_pb2

    from flyte._internal.controllers.remote._action import Action

    return Action(
        parent_action_name=parent,
        action_id=identifier_pb2.ActionIdentifier(
            name="child", run=identifier_pb2.RunIdentifier(org="acme", project="p", domain="development", name=run_name)
        ),
        type="trace",
    )


def _action_of(headers: dict):
    return decode_credential(headers).get("action")


class TestWorkloadAction:
    """A pod workload (a reusable worker running many actions) names, in its credential, the action of each request."""

    @pytest.fixture
    def pod_workload(self, pod):
        env, _ = pod
        env.setenv("_U_WORKLOAD", POD_WORKLOAD)
        return WorkloadTokenAuthenticator(endpoint="dns:///acme.union.ai:443")

    @pytest.mark.asyncio
    async def test_controller_enqueue_names_the_parent_action(self, pod_workload):
        """The controller's Actions service calls already carry x-actions-*; the action is taken from them."""
        controller, sent = _launch_controller(pod_workload, actions_service=True)
        await controller._bg_launch(_child(run_name="r1", parent="a0"))
        assert decode_credential(sent[0]) == {
            "token": "projected-sa-token-1",
            "workload": POD_WORKLOAD,
            "action": {"run": "r1", "name": "a0"},
        }

    @pytest.mark.asyncio
    async def test_controller_call_names_its_own_action_not_the_callers_context(self, pod_workload):
        """The controller's calls name the action in hand, even when another action's task context is current."""
        from flyte._context import internal_ctx

        controller, sent = _launch_controller(pod_workload, actions_service=True)
        with internal_ctx().replace_task_context(_task_context("other-run", "other-action")):
            await controller._bg_launch(_child(run_name="r1", parent="a0"))
        assert _action_of(sent[0]) == {"run": "r1", "name": "a0"}

    @pytest.mark.asyncio
    async def test_a_call_inside_a_running_task_names_its_action(self, pod_workload):
        from flyte._context import internal_ctx

        with internal_ctx().replace_task_context(_task_context("r2", "a3")):
            headers = await _send(pod_workload)
        assert _action_of(headers) == {"run": "r2", "name": "a3"}

    @pytest.mark.asyncio
    async def test_a_call_inside_a_trace_names_the_running_task(self, pod_workload):
        from flyte._context import internal_ctx
        from flyte.models import ActionID

        tctx = _task_context("r2", "a3")
        in_trace = tctx.replace(action=ActionID(name="trace-1", run_name="r2", project="p", domain="development"))
        with internal_ctx().replace_task_context(in_trace):
            headers = await _send(pod_workload)
        assert _action_of(headers) == {"run": "r2", "name": "a3"}

    @pytest.mark.asyncio
    async def test_no_action_without_a_known_action(self, pod_workload):
        headers = await _send(pod_workload, {"x-actions-run": "r1"})  # no parent action: not a known action
        assert set(headers) == {"x-actions-run", "authorization"}
        assert decode_credential(headers) == {"token": "projected-sa-token-1", "workload": POD_WORKLOAD}

    @pytest.mark.asyncio
    async def test_each_request_gets_its_own_credential(self, pod_workload, pod):
        """The action differs per request, and a rotated token is picked up, without a cached credential."""
        _, token = pod
        first = await _send(pod_workload, {"x-actions-run": "r1", "x-actions-parent-action": "a0"})
        second = await _send(pod_workload, {"x-actions-run": "r9", "x-actions-parent-action": "a7"})
        assert _action_of(first) == {"run": "r1", "name": "a0"}
        assert _action_of(second) == {"run": "r9", "name": "a7"}
        token.write_text("projected-sa-token-2")
        await pod_workload.refresh_credentials()
        assert sent_token(await _send(pod_workload)) == "projected-sa-token-2"

    @pytest.mark.asyncio
    async def test_a_retry_replaces_the_credential(self, pod_workload):
        from connectrpc.errors import ConnectError

        from flyte.remote._client.auth._interceptors.auth import AuthUnaryInterceptor

        sent = []

        async def call_next(request, ctx):
            sent.append(dict(ctx.request_headers()))
            if len(sent) == 1:
                raise ConnectError(Code.UNKNOWN, "transient")
            return "ok"

        ctx = _Ctx({"x-actions-run": "r1", "x-actions-parent-action": "a0"})
        await AuthUnaryInterceptor(lambda: pod_workload).intercept_unary(call_next, object(), ctx)
        assert [_action_of(h) for h in sent] == [{"run": "r1", "name": "a0"}] * 2
        assert all(set(h) == {"x-actions-run", "x-actions-parent-action", "authorization"} for h in sent)

    @pytest.mark.asyncio
    async def test_the_fallback_api_key_carries_no_credential(self, pod_workload):
        factory, _, _ = TestWorkloadTokenAuthenticator._fallback_stub()
        pod_workload._fallback_factory = factory
        first = await pod_workload.get_auth_headers()
        pod_workload.note_refusal(Code.UNAUTHENTICATED, first.creds_id)
        await pod_workload.refresh_credentials(first.creds_id)
        headers = await _send(pod_workload, {"x-actions-run": "r1", "x-actions-parent-action": "a0"})
        assert headers["authorization"] == "Bearer api-key"
        assert not any(k.startswith("x-union-workload") for k in headers)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("workload", [WORKLOAD, "app/acme/p/development/my-app"], ids=["attempt", "app"])
    async def test_no_action_for_attempt_and_app_workloads(self, pod, workload):
        from flyte._context import internal_ctx

        env, _ = pod
        env.setenv("_U_WORKLOAD", workload)
        auth = WorkloadTokenAuthenticator(endpoint="dns:///acme.union.ai:443")

        controller, sent = _launch_controller(auth, actions_service=True)
        await controller._bg_launch(_child())
        with internal_ctx().replace_task_context(_task_context("r2", "a3")):
            sent.append(await _send(auth))
        for headers in sent:
            assert decode_credential(headers) == {"token": "projected-sa-token-1", "workload": workload}


def test_encoding_is_an_unsigned_compact_jws():
    """The exact encoding, so other implementations (the Rust controller) can be checked against it."""
    from flyte._workload_credential import encode_workload_credential

    attempt = encode_workload_credential("tok", WORKLOAD)
    assert attempt == (
        "eyJ0eXAiOiJ1bmlvbi13b3JrbG9hZCIsImFsZyI6Im5vbmUifQ."
        + base64.urlsafe_b64encode(b'{"token":"tok","workload":"acme/p/development/r1/a0/1"}').rstrip(b"=").decode()
        + "."
    )
    pod = encode_workload_credential("tok", POD_WORKLOAD, ("r1", "a0"))
    payload = json.loads(_b64decode(pod.split(".")[1]))
    assert payload == {"token": "tok", "workload": POD_WORKLOAD, "action": {"run": "r1", "name": "a0"}}
