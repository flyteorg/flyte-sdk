"""
Tests for the MCP secret tools: values never pass through the model.

Covers:
- Schema: `create_secret`/`update_secret` take no ``value`` unless ``allow_inline_secret_values``.
- The one-time-URL flow end to end through the env's Starlette app: mint a URL from a tool call,
  GET the form, POST a value, and see the SDK called once under the right project/domain.
- The value never shows up in a tool result, a response body or a log record.
- Replay, expiry and unknown tokens are refused without calling the SDK.
- URL elicitation for clients that advertise it, a plain result for those that don't, and the CLI
  fallback over stdio.
- ``get_secret`` against real ``definition_pb2.Secret`` protos.
- ``generate_secret``: the stored private key parses, matches the returned public key and
  fingerprint, and appears in no tool result or log record.
- Ingress validators: a rejected value leaves the token usable and never reaches the SDK; a validator
  that raises or times out is a rejection; the parameter is absent from the schema when none exist.
- Scope safety: no secret tool runs without a resolvable project/domain.

Every remote call is monkeypatched; no test touches the network.
"""

from __future__ import annotations

import asyncio
import base64
import builtins
import hashlib
import json
import logging
import pathlib
import struct
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
from flyteidl2.common import identifier_pb2
from flyteidl2.secret import definition_pb2
from google.protobuf import timestamp_pb2
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.shared.exceptions import McpError, UrlElicitationRequiredError
from mcp.types import ElicitResult
from starlette.requests import Request
from starlette.testclient import TestClient

import flyte
import flyte._initialize as init_mod
import flyte.remote
from flyte.ai.mcp import FlyteMCPAppEnvironment, _secret_input
from flyte.ai.mcp._flyte_mcp_app import DOMAIN_ENV_VAR, PROJECT_ENV_VAR, ToolError

VALUE = "hunter2-very-secret-value"
AUTH = {"Authorization": "Bearer test-token"}
MOUNT = "/flyte-mcp"

SECRET_TOOLS = ["list_secrets", "get_secret", "create_secret", "update_secret", "generate_secret", "delete_secret"]


def _fn(env: FlyteMCPAppEnvironment, name: str):
    return env._mcp_server._tool_manager._tools[name].fn


@pytest.fixture(autouse=True)
def _flyte_logs_reach_caplog(monkeypatch, caplog):
    # The "flyte" logger does not propagate to the root logger, so without this every
    # `x not in caplog.text` check below would pass no matter what was logged. Its level is set
    # too: other suites (the CLI's) leave it raised, which would filter records out before caplog.
    monkeypatch.setattr(logging.getLogger("flyte"), "propagate", True)
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="flyte")


@pytest.fixture
def init_config(monkeypatch):
    cfg = init_mod._InitConfig(
        root_dir=pathlib.Path.cwd(),
        org="test-org",
        project="cfg-project",
        domain="cfg-domain",
        client=object(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(init_mod, "_init_config", cfg)
    return cfg


class _Recorder:
    """Stands in for a syncified ``Secret`` classmethod; records the scope each call ran under."""

    def __init__(self, exc: Exception | None = None):
        self.calls: list[dict] = []
        self.exc = exc

    async def aio(self, **kwargs):
        cfg = init_mod._get_init_config()
        self.calls.append({**kwargs, "scope": (cfg.project, cfg.domain)})
        if self.exc is not None:
            raise self.exc


class _Existence:
    """Stands in for ``Secret.get`` as the create/update existence check sees it: found or NOT_FOUND."""

    def __init__(self, exists: bool = False, exc: Exception | None = None):
        self.exists, self.exc, self.calls = exists, exc, []

    async def aio(self, *args, **kwargs):
        cfg = init_mod._get_init_config()
        self.calls.append({"args": args, "scope": (cfg.project, cfg.domain)})
        if self.exc is not None:
            raise self.exc
        if not self.exists:
            raise ConnectError(Code.NOT_FOUND, "no such secret")
        return SimpleNamespace(name=args[0])


@pytest.fixture(autouse=True)
def secret_get(monkeypatch):
    """`Secret.get` as the existence checks see it. Defaults to missing, which is what create needs;
    tests of update (or of create over an existing secret) set ``secret_get.exists = True``."""
    getter = _Existence()
    monkeypatch.setattr(flyte.remote.Secret, "get", getter)
    return getter


def _ctx(
    *,
    url_elicitation: bool = False,
    mount: str = MOUNT,
    host: str = "testserver",
    prefix: str = "",
    inner: str = "",
    app_root_path: bool = True,
):
    """A tool-call context shaped like FastMCP's: the request seen from *inside* the MCP mount.

    `inner` is a second mount below the MCP one (SSE's `/messages`). `app_root_path=False` is a
    request that was not routed through a Mount at all.
    """
    scope = {
        "type": "http",
        "method": "POST",
        "scheme": "http",
        "server": (host, 80),
        "headers": [(b"host", host.encode())],
        "path": f"{prefix}{mount}{inner}/mcp",
        "root_path": f"{prefix}{mount}{inner}",
        "query_string": b"",
    }
    if app_root_path:
        # Starlette's Mount sets this to the root_path from before the first mount.
        scope["app_root_path"] = prefix
    request = Request(scope)
    url_cap = SimpleNamespace() if url_elicitation else None
    capabilities = SimpleNamespace(elicitation=SimpleNamespace(form=None, url=url_cap))
    return SimpleNamespace(
        request_context=SimpleNamespace(request=request),
        session=SimpleNamespace(client_params=SimpleNamespace(capabilities=capabilities)),
    )


def _token_of(url: str) -> str:
    assert url.startswith("http://testserver/secret-input/"), url
    return url.rsplit("/", 1)[1]


def _path_of(url: str) -> str:
    return "/" + url.split("/", 3)[3]


# ------------------------------
# Schema
# ------------------------------


class TestInputSchema:
    @pytest.mark.parametrize("tool", ["create_secret", "update_secret"])
    def test_value_is_absent_by_default(self, tool):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        schema = env._mcp_server._tool_manager._tools[tool].parameters
        assert "value" not in schema["properties"]
        assert "name" in schema["required"]

    @pytest.mark.parametrize("tool", ["create_secret", "update_secret"])
    def test_value_is_required_when_opted_in(self, tool):
        env = FlyteMCPAppEnvironment(name="test-mcp", allow_inline_secret_values=True)
        schema = env._mcp_server._tool_manager._tools[tool].parameters
        assert "value" in schema["required"]

    def test_opt_in_shows_in_rich_repr(self):
        assert "allow_inline_secret_values" not in dict(FlyteMCPAppEnvironment(name="t").__rich_repr__())
        env = FlyteMCPAppEnvironment(name="t", allow_inline_secret_values=True)
        assert dict(env.__rich_repr__())["allow_inline_secret_values"] is True

    @pytest.mark.asyncio
    async def test_opt_in_update_calls_the_sdk_with_the_value(self, init_config, monkeypatch, secret_get):
        secret_get.exists = True
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "update", recorder)
        env = FlyteMCPAppEnvironment(name="test-mcp", allow_inline_secret_values=True)

        result = await _fn(env, "update_secret")(name="s", value="v", project="p", domain="d")

        assert result == {"name": "s", "updated": True}
        assert recorder.calls == [{"name": "s", "value": "v", "scope": ("p", "d")}]


# ------------------------------
# The one-time-URL flow
# ------------------------------


class TestFormFlow:
    @pytest.fixture
    def env(self, init_config):
        return FlyteMCPAppEnvironment(name="test-mcp")

    @pytest.fixture
    def client(self, env):
        return TestClient(env._starlette_app, headers=AUTH, raise_server_exceptions=False)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("op", "tool"), [("create", "create_secret"), ("update", "update_secret")])
    async def test_value_goes_browser_to_sdk_only(self, env, client, monkeypatch, caplog, secret_get, op, tool):
        secret_get.exists = op == "update"
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, op, recorder)
        caplog.set_level(logging.DEBUG)

        result = await _fn(env, tool)(name="db-pass", project="p", domain="d", ctx=_ctx())

        assert result["status"] == "awaiting_value"
        assert (result["op"], result["name"], result["project"], result["domain"]) == (op, "db-pass", "p", "d")
        assert result["cli_fallback"]
        assert "Do NOT ask" in result["message"]
        # The route is on the outer app, not under the MCP mount.
        assert result["url"].startswith("http://testserver/secret-input/")
        assert MOUNT not in result["url"]

        form = client.get(_path_of(result["url"]))
        assert form.status_code == 200
        assert form.headers["cache-control"] == "no-store"
        assert form.headers["referrer-policy"] == "no-referrer"
        assert 'type="password"' in form.text or "type=password" in form.text
        assert "<textarea" in form.text
        assert "db-pass" in form.text and ">p<" in form.text and ">d<" in form.text
        assert "<script" not in form.text and "http" not in form.text.replace("http-equiv", "")

        posted = client.post(
            _path_of(result["url"]),
            data={"value": VALUE},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert posted.status_code == 200
        assert posted.headers["cache-control"] == "no-store"
        assert recorder.calls == [{"name": "db-pass", "value": VALUE, "scope": ("p", "d")}]

        # Nowhere the model or an operator could read it.
        assert VALUE not in str(result)
        assert VALUE not in form.text
        assert VALUE not in posted.text
        assert VALUE not in caplog.text

    def test_multiline_value_is_normalised_to_lf(self, env, client, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)
        token, _ = env._secret_inputs.mint("create", "pem", "p", "d")

        response = client.post(f"/secret-input/{token}", data={"value": "", "value_multiline": "a\r\nb\r\n"})

        assert response.status_code == 200
        assert recorder.calls[0]["value"] == "a\nb\n"

    @pytest.mark.asyncio
    async def test_concurrent_posts_yield_one_write_and_one_410(self, env, monkeypatch):
        # The stubbed existence check parks the first POST; the second must be refused
        # while the first is still in flight, not after it finishes.
        gate, entered = asyncio.Event(), asyncio.Event()

        async def slow_get(name):
            entered.set()
            await gate.wait()
            raise ConnectError(Code.NOT_FOUND, "x")

        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "get", SimpleNamespace(aio=slow_get))
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)
        token, _ = env._secret_inputs.mint("create", "s", "p", "d")

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=env._starlette_app), base_url="http://t") as c:
            first = asyncio.create_task(c.post(f"/secret-input/{token}", data={"value": "one"}, headers=AUTH))
            await entered.wait()
            # The timeout only turns a regression (the second POST parks on the gate too) into a failure.
            second = await asyncio.wait_for(c.post(f"/secret-input/{token}", data={"value": "two"}, headers=AUTH), 10)
            assert second.status_code == 410
            assert (await c.get(f"/secret-input/{token}", headers=AUTH)).status_code == 410
            gate.set()
            assert (await first).status_code == 200

        assert recorder.calls == [{"name": "s", "value": "one", "scope": ("p", "d")}]

    def test_chunked_body_over_the_limit_is_refused_without_content_length(self, env, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)
        token, pending = env._secret_inputs.mint("create", "s", "p", "d")

        async def body():
            yield b"value="
            yield b"x" * _secret_input.MAX_FORM_BYTES

        async def post():
            transport = httpx.ASGITransport(app=env._starlette_app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                return await c.post(f"/secret-input/{token}", content=body(), headers=AUTH)

        response = asyncio.run(post())

        assert response.status_code == 413
        assert recorder.calls == []
        assert not pending.in_flight  # released: the link was not burned

    @pytest.mark.parametrize("fields", [{}, {"value": ""}, {"value": "a", "value_multiline": "b"}])
    def test_missing_or_ambiguous_value_is_rejected_and_token_stays_valid(self, env, client, monkeypatch, fields):
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)
        token, pending = env._secret_inputs.mint("create", "s", "p", "d")

        response = client.post(f"/secret-input/{token}", data=fields)

        assert response.status_code == 400
        assert recorder.calls == []
        assert pending.used is False

    def test_second_post_is_refused_and_sdk_not_called_again(self, env, client, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)
        token, _ = env._secret_inputs.mint("create", "s", "p", "d")

        assert client.post(f"/secret-input/{token}", data={"value": VALUE}).status_code == 200
        again = client.post(f"/secret-input/{token}", data={"value": VALUE})

        assert again.status_code == 410
        assert client.get(f"/secret-input/{token}").status_code == 410
        assert len(recorder.calls) == 1
        assert VALUE not in again.text

    def test_expired_token_is_refused(self, env, client, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)
        now = [1_000.0]
        monkeypatch.setattr(_secret_input, "_now", lambda: now[0])
        token, _ = env._secret_inputs.mint("create", "s", "p", "d")

        now[0] += _secret_input.TOKEN_TTL_S + 1

        assert client.get(f"/secret-input/{token}").status_code == 410
        assert client.post(f"/secret-input/{token}", data={"value": VALUE}).status_code == 410
        assert recorder.calls == []

    def test_unknown_token_is_404(self, client, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)

        assert client.get("/secret-input/nope").status_code == 404
        assert client.post("/secret-input/nope", data={"value": VALUE}).status_code == 404
        assert recorder.calls == []

    def test_sdk_error_is_shown_without_the_value_and_burns_the_token(self, env, client, monkeypatch, caplog):
        recorder = _Recorder(exc=RuntimeError(f"AlreadyExists: could not store {VALUE}"))
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)
        token, _ = env._secret_inputs.mint("create", "s", "p", "d")

        response = client.post(f"/secret-input/{token}", data={"value": VALUE})

        assert response.status_code == 502
        assert "RuntimeError" in response.text and "AlreadyExists" in response.text
        assert VALUE not in response.text
        assert VALUE not in caplog.text
        assert client.post(f"/secret-input/{token}", data={"value": VALUE}).status_code == 410
        assert len(recorder.calls) == 1

    def test_token_is_marked_used_before_the_sdk_call(self, env, client, monkeypatch):
        token, pending = env._secret_inputs.mint("create", "s", "p", "d")
        seen = []

        class _Checks(_Recorder):
            async def aio(self, **kwargs):
                seen.append(pending.used)

        monkeypatch.setattr(flyte.remote.Secret, "create", _Checks())
        client.post(f"/secret-input/{token}", data={"value": VALUE})

        assert seen == [True]

    def test_requests_without_credentials_are_rejected_by_the_auth_middleware(self, env, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)
        token, _ = env._secret_inputs.mint("create", "s", "p", "d")
        anonymous = TestClient(env._starlette_app, raise_server_exceptions=False)

        assert anonymous.get(f"/secret-input/{token}").status_code == 401
        assert anonymous.post(f"/secret-input/{token}", data={"value": VALUE}).status_code == 401
        assert recorder.calls == []

    def test_oversized_body_is_refused(self, env, client, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)
        token, _ = env._secret_inputs.mint("create", "s", "p", "d")

        response = client.post(f"/secret-input/{token}", data={"value": "x" * (_secret_input.MAX_FORM_BYTES + 1)})

        assert response.status_code == 413
        assert recorder.calls == []

    def test_expired_tokens_are_swept_when_minting(self, env, monkeypatch):
        now = [1_000.0]
        monkeypatch.setattr(_secret_input, "_now", lambda: now[0])
        old, _ = env._secret_inputs.mint("create", "a", "p", "d")
        now[0] += _secret_input.TOKEN_TTL_S + 1
        env._secret_inputs.mint("create", "b", "p", "d")

        assert env._secret_inputs.get(old) is None

    def test_no_route_when_no_secret_write_tool_is_enabled(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", tools=["whoami"])
        client = TestClient(env._starlette_app, headers=AUTH, raise_server_exceptions=False)
        assert client.get("/secret-input/x").status_code == 404

    def test_no_route_in_read_only_mode(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", read_only=True)
        client = TestClient(env._starlette_app, headers=AUTH, raise_server_exceptions=False)
        assert client.get("/secret-input/x").status_code == 404


class TestTokens:
    def test_tokens_are_unguessable_and_unique(self):
        store = _secret_input.SecretInputStore()
        tokens = {store.mint("create", "s", "p", "d")[0] for _ in range(50)}
        assert len(tokens) == 50
        assert all(len(t) >= 43 for t in tokens)  # secrets.token_urlsafe(32)

    def test_expiry_is_ten_minutes_out(self, monkeypatch):
        monkeypatch.setattr(_secret_input, "_now", lambda: 100.0)
        _, pending = _secret_input.SecretInputStore().mint("update", "s", "p", "d")
        assert pending.expires_at == 700.0
        assert (pending.op, pending.used) == ("update", False)


# ------------------------------
# What the tool returns, per client / transport
# ------------------------------


class TestAwaitingValueResult:
    @pytest.fixture
    def env(self, init_config):
        return FlyteMCPAppEnvironment(name="test-mcp")

    @pytest.mark.asyncio
    async def test_url_elicitation_capable_client_gets_an_elicitation_error(self, env):
        with pytest.raises(UrlElicitationRequiredError) as exc_info:
            await _fn(env, "create_secret")(name="s", project="p", domain="d", ctx=_ctx(url_elicitation=True))

        (params,) = exc_info.value.elicitations
        assert params.mode == "url"
        assert params.url.startswith("http://testserver/secret-input/")
        assert params.elicitationId == params.url.rsplit("/", 1)[1]
        assert env._secret_inputs.get(params.elicitationId) is not None
        assert "s" in params.message and "p/d" in params.message

    @pytest.mark.asyncio
    async def test_client_without_url_capability_gets_a_result(self, env, secret_get):
        secret_get.exists = True
        result = await _fn(env, "update_secret")(name="s", project="p", domain="d", ctx=_ctx())
        assert result["status"] == "awaiting_value"
        assert env._secret_inputs.get(_token_of(result["url"])).op == "update"
        assert result["expires_at"]

    @pytest.mark.asyncio
    async def test_client_that_reports_no_capabilities_gets_a_result(self, env):
        ctx = _ctx()
        ctx.session.client_params.capabilities.elicitation = None
        assert (await _fn(env, "create_secret")(name="s", project="p", domain="d", ctx=ctx))["url"]

    @pytest.mark.asyncio
    async def test_proxy_prefix_is_kept_and_mount_is_dropped(self, env):
        result = await _fn(env, "create_secret")(name="s", project="p", domain="d", ctx=_ctx(prefix="/proxy"))
        assert result["url"].startswith("http://testserver/proxy/secret-input/")

    @pytest.mark.asyncio
    async def test_url_from_inside_a_nested_mount_has_only_the_proxy_prefix(self, env):
        # SSE tool calls arrive via a mount nested in the MCP mount (`/messages`).
        ctx = _ctx(prefix="/proxy", inner="/messages")
        result = await _fn(env, "create_secret")(name="s", project="p", domain="d", ctx=ctx)
        assert result["url"].startswith("http://testserver/proxy/secret-input/")

    @pytest.mark.asyncio
    async def test_custom_mount_path(self, init_config):
        env = FlyteMCPAppEnvironment(name="test-mcp", mcp_mount_path="/m")
        result = await _fn(env, "create_secret")(name="s", project="p", domain="d", ctx=_ctx(mount="/m"))
        assert result["url"].startswith("http://testserver/secret-input/")

    @pytest.mark.asyncio
    async def test_no_request_context_falls_back_to_cli(self, env):
        result = await _fn(env, "create_secret")(name="s", project="p", domain="d", ctx=None)
        assert result["status"] == "awaiting_value"
        assert "url" not in result
        assert result["cli_fallback"] == "flyte create secret s -p p -d d"
        assert env._secret_inputs._pending == {}

    @pytest.mark.asyncio
    async def test_cli_fallback_quotes_hostile_names(self, init_config):
        env = FlyteMCPAppEnvironment(name="test-mcp", transport="stdio")

        result = await _fn(env, "create_secret")(name="x; rm -rf ~", project="p", domain="d", ctx=None)

        assert result["cli_fallback"] == "flyte create secret 'x; rm -rf ~' -p p -d d"

    @pytest.mark.asyncio
    async def test_stdio_returns_cli_fallback_and_mints_nothing(self, init_config, secret_get):
        env = FlyteMCPAppEnvironment(name="test-mcp", transport="stdio")

        created = await _fn(env, "create_secret")(name="s", project="p", domain="d", ctx=_ctx(url_elicitation=True))
        secret_get.exists = True
        updated = await _fn(env, "update_secret")(name="s", project="p", domain="d", ctx=_ctx(url_elicitation=True))

        assert created["status"] == updated["status"] == "awaiting_value"
        assert created["cli_fallback"] == "flyte create secret s -p p -d d"
        # `flyte create secret` replaces an existing secret's value, so update uses the same command.
        # Deleting first would leave the secret missing if the prompt were aborted.
        assert updated["cli_fallback"] == "flyte create secret s -p p -d d"
        assert "url" not in created and "url" not in updated
        assert env._secret_inputs._pending == {}


class TestOverRealStreamableHttp:
    def test_url_comes_back_through_the_mounted_mcp_endpoint(self, init_config):
        # Exercises the real request object FastMCP hands the tool (not a fake ctx): inside the
        # mount its root_path includes the mount, which the URL must not.
        headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        env = FlyteMCPAppEnvironment(name="test-mcp", requires_auth=False)
        call = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "create_secret", "arguments": {"name": "abc", "project": "p", "domain": "d"}},
        }

        with TestClient(env._starlette_app, base_url="http://example.org:8080") as client:
            response = client.post(f"{MOUNT}/mcp", headers=headers, json=call)
            assert response.status_code == 200
            result = json.loads(response.json()["result"]["content"][0]["text"])
            assert result["url"].startswith("http://example.org:8080/secret-input/")
            assert client.get(_path_of(result["url"])).status_code == 200


@asynccontextmanager
async def _served(env: FlyteMCPAppEnvironment):
    """Serve `env` on a loopback port the OS picks; yields the base URL. No network beyond lo."""
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(env._starlette_app, host="127.0.0.1", port=0, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    try:
        # uvicorn offers no event for this; poll until the socket is bound, for at most 10 s.
        for _ in range(1000):
            if server.started:
                break
            await asyncio.sleep(0.01)
        else:
            raise RuntimeError("uvicorn did not start within 10 s")
        yield f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    finally:
        server.should_exit = True
        await serving


class TestOverRealSse:
    """SSE nests a mount (`/messages`) inside the MCP mount; `TestClient` cannot hold the stream open."""

    @pytest.fixture
    def env(self, init_config):
        return FlyteMCPAppEnvironment(name="test-mcp", transport="sse", requires_auth=False)

    async def _assert_form_works(self, base, url, recorder):
        assert url.startswith(f"{base}/secret-input/"), url
        async with httpx.AsyncClient() as http:
            assert (await http.get(url)).status_code == 200
            assert (await http.post(url, data={"value": VALUE})).status_code == 200
        assert recorder.calls == [{"name": "abc", "value": VALUE, "scope": ("p", "d")}]

    @pytest.mark.asyncio
    async def test_plain_client_gets_a_url_that_serves_and_writes(self, env, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)

        async with _served(env) as base:
            async with sse_client(f"{base}{MOUNT}/sse") as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("create_secret", {"name": "abc", "project": "p", "domain": "d"})
            await self._assert_form_works(base, json.loads(result.content[0].text)["url"], recorder)  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_url_elicitation_client_gets_a_url_that_serves_and_writes(self, env, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)

        async def refuse(context, params):
            return ElicitResult(action="decline")

        async with _served(env) as base:
            async with sse_client(f"{base}{MOUNT}/sse") as (read, write):
                async with ClientSession(read, write, elicitation_callback=refuse) as session:
                    await session.initialize()
                    with pytest.raises(McpError) as exc_info:
                        await session.call_tool("create_secret", {"name": "abc", "project": "p", "domain": "d"})
            assert exc_info.value.error.code == -32042
            (elicitation,) = exc_info.value.error.data["elicitations"]
            await self._assert_form_works(base, elicitation["url"], recorder)


# ------------------------------
# get_secret
# ------------------------------


def _secret_pb(overall, presence: dict[str, int], name="s") -> definition_pb2.Secret:
    created = timestamp_pb2.Timestamp()
    created.FromJsonString("2026-01-02T03:04:05Z")
    return definition_pb2.Secret(
        id=definition_pb2.SecretIdentifier(organization="o", project="p", domain="d", name=name),
        secret_metadata=definition_pb2.SecretMetadata(
            created_time=created,
            type=definition_pb2.SecretType.SECRET_TYPE_GENERIC,
            secret_status=definition_pb2.SecretStatus(
                overall_status=overall,
                cluster_status=[
                    definition_pb2.ClusterSecretStatus(
                        cluster=identifier_pb2.ClusterIdentifier(name=cluster), presence_status=status
                    )
                    for cluster, status in presence.items()
                ],
            ),
        ),
    )


class _Getter:
    def __init__(self, result=None, exc: Exception | None = None):
        self.result, self.exc, self.calls = result, exc, []

    async def aio(self, *args, **kwargs):
        cfg = init_mod._get_init_config()
        self.calls.append({"args": args, "kwargs": kwargs, "scope": (cfg.project, cfg.domain)})
        if self.exc is not None:
            raise self.exc
        return self.result


class TestGetSecret:
    @pytest.fixture
    def env(self, init_config):
        return FlyteMCPAppEnvironment(name="test-mcp")

    @pytest.mark.asyncio
    async def test_fully_present_is_ready(self, env, monkeypatch):
        pb = _secret_pb(
            definition_pb2.OverallStatus.FULLY_PRESENT,
            {"c1": definition_pb2.SecretPresenceStatus.PRESENT, "c2": definition_pb2.SecretPresenceStatus.PRESENT},
        )
        getter = _Getter(flyte.remote.Secret(pb2=pb))
        monkeypatch.setattr(flyte.remote.Secret, "get", getter)

        result = await _fn(env, "get_secret")(name="s", project="p", domain="d")

        assert result == {
            "name": "s",
            "project": "p",
            "domain": "d",
            "exists": True,
            "type": "regular",
            "created_time": "2026-01-02T03:04:05+00:00",
            "overall_status": "FULLY_PRESENT",
            "cluster_status": [
                {"cluster": "c1", "presence": "PRESENT"},
                {"cluster": "c2", "presence": "PRESENT"},
            ],
            "ready": True,
        }
        assert getter.calls == [{"args": ("s",), "kwargs": {}, "scope": ("p", "d")}]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "overall",
        [
            definition_pb2.OverallStatus.PARTIALLY_PRESENT,
            definition_pb2.OverallStatus.UNSPECIFIED,
            definition_pb2.OverallStatus.UNKNOWN_STATUS,
        ],
    )
    async def test_anything_but_fully_present_is_not_ready(self, env, monkeypatch, overall):
        pb = _secret_pb(overall, {"c1": definition_pb2.SecretPresenceStatus.MISSING})
        monkeypatch.setattr(flyte.remote.Secret, "get", _Getter(flyte.remote.Secret(pb2=pb)))

        result = await _fn(env, "get_secret")(name="s", project="p", domain="d")

        assert result["ready"] is False
        assert result["cluster_status"] == [{"cluster": "c1", "presence": "MISSING"}]

    @pytest.mark.asyncio
    async def test_missing_secret_reports_exists_false(self, env, monkeypatch):
        monkeypatch.setattr(flyte.remote.Secret, "get", _Getter(exc=ConnectError(Code.NOT_FOUND, "no such secret")))

        result = await _fn(env, "get_secret")(name="s", project="p", domain="d")

        assert result == {"name": "s", "project": "p", "domain": "d", "exists": False, "ready": False}

    @pytest.mark.asyncio
    async def test_other_errors_propagate(self, env, monkeypatch):
        monkeypatch.setattr(flyte.remote.Secret, "get", _Getter(exc=ConnectError(Code.PERMISSION_DENIED, "no")))

        with pytest.raises(ConnectError):
            await _fn(env, "get_secret")(name="s", project="p", domain="d")

    def test_docstring_points_at_run_task_for_content_checks(self, env):
        doc = _fn(env, "get_secret").__doc__
        assert "run_task" in doc

    def test_annotations(self, env):
        ann = env._mcp_server._tool_manager._tools["get_secret"].annotations
        assert ann.readOnlyHint is True and ann.idempotentHint is True


# ------------------------------
# Scope safety: never org-wide
# ------------------------------


class TestScopeSafety:
    """With nothing to resolve a project/domain from, no secret tool may reach the SDK."""

    @pytest.fixture(params=[False, True], ids=["default", "inline"])
    def env(self, request, monkeypatch):
        monkeypatch.delenv(PROJECT_ENV_VAR, raising=False)
        monkeypatch.delenv(DOMAIN_ENV_VAR, raising=False)
        monkeypatch.setattr(init_mod, "_init_config", None)
        return FlyteMCPAppEnvironment(name="test-mcp", allow_inline_secret_values=request.param)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool", SECRET_TOOLS)
    async def test_unresolvable_scope_raises_before_any_sdk_call(self, env, monkeypatch, tool):
        recorders = {}
        for method in ("listall", "get", "create", "update", "delete"):
            recorders[method] = _Recorder()
            monkeypatch.setattr(flyte.remote.Secret, method, recorders[method])
        kwargs = {"name": "s"}
        if tool in ("create_secret", "update_secret") and env.allow_inline_secret_values:
            kwargs["value"] = "v"
        if tool == "list_secrets":
            kwargs = {}

        with pytest.raises(ToolError, match="Missing project and domain"):
            await _fn(env, tool)(**kwargs, ctx=_ctx())

        assert all(r.calls == [] for r in recorders.values())
        assert env._secret_inputs._pending == {}

    @pytest.mark.asyncio
    async def test_only_one_half_resolvable_is_still_refused(self, env, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "delete", recorder)

        with pytest.raises(ToolError, match="Missing domain"):
            await _fn(env, "delete_secret")(name="s", project="p")
        assert recorder.calls == []


class TestRequestsCarryResolvedScope:
    @pytest.mark.asyncio
    async def test_env_defaults_and_arguments_both_resolve(self, init_config, monkeypatch):
        monkeypatch.setenv(PROJECT_ENV_VAR, "env-project")
        getter = _Getter(flyte.remote.Secret(pb2=_secret_pb(definition_pb2.OverallStatus.FULLY_PRESENT, {})))
        monkeypatch.setattr(flyte.remote.Secret, "get", getter)
        env = FlyteMCPAppEnvironment(name="test-mcp")

        await _fn(env, "get_secret")(name="s", domain="arg-domain")
        await _fn(env, "get_secret")(name="s", project="arg-project")

        assert [c["scope"] for c in getter.calls] == [("env-project", "arg-domain"), ("arg-project", "cfg-domain")]

    def test_cluster_pool_is_not_exposed(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", allow_inline_secret_values=True)
        for tool in SECRET_TOOLS:
            assert "cluster_pool" not in env._mcp_server._tool_manager._tools[tool].parameters["properties"]


# ------------------------------
# generate_secret
# ------------------------------


class _Capture(_Recorder):
    """A ``Secret.create`` stand-in that keeps the value it was given."""

    @property
    def value(self) -> str:
        (call,) = self.calls
        return call["value"]


def _openssh_fingerprint_ed25519(public_key: ed25519.Ed25519PublicKey) -> str:
    # Built by hand from the wire format (RFC 4253 section 6.6) rather than via the code under test.
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    blob = struct.pack(">I", 11) + b"ssh-ed25519" + struct.pack(">I", len(raw)) + raw
    return "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")


class TestGenerateSecret:
    @pytest.fixture
    def env(self, init_config):
        return FlyteMCPAppEnvironment(name="test-mcp")

    @pytest.mark.asyncio
    async def test_ed25519_is_the_default_and_only_the_public_half_comes_back(self, env, monkeypatch, caplog):
        recorder = _Capture()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)
        caplog.set_level(logging.DEBUG)

        result = await _fn(env, "generate_secret")(name="deploy-key", project="p", domain="d")

        assert set(result) == {"name", "project", "domain", "kind", "public_key", "fingerprint", "created", "message"}
        assert (result["name"], result["project"], result["domain"]) == ("deploy-key", "p", "d")
        assert result["kind"] == "ssh-ed25519"
        assert result["created"] is True
        assert "private" in result["message"] and "public_key" in result["message"]

        # The stored value is an unencrypted OpenSSH private key whose public half is the one returned.
        private_pem = recorder.value
        assert recorder.calls[0]["name"] == "deploy-key" and recorder.calls[0]["scope"] == ("p", "d")
        assert private_pem.startswith("-----BEGIN OPENSSH PRIVATE KEY-----")
        private_key = serialization.load_ssh_private_key(private_pem.encode(), password=None)
        assert isinstance(private_key, ed25519.Ed25519PrivateKey)
        openssh_line = (
            private_key.public_key()
            .public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH)
            .decode()
        )
        assert result["public_key"] == f"{openssh_line} deploy-key"
        assert result["fingerprint"] == _openssh_fingerprint_ed25519(private_key.public_key())

        # Nowhere the model or an operator could read it.
        body = private_pem.split("-----")[2].strip()
        assert body not in json.dumps(result)
        assert body not in caplog.text

    @pytest.mark.asyncio
    async def test_rsa_is_4096_bits(self, env, monkeypatch):
        recorder = _Capture()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)

        result = await _fn(env, "generate_secret")(
            name="k", kind="ssh-rsa", comment="ci@example.org", project="p", domain="d"
        )

        private_key = serialization.load_ssh_private_key(recorder.value.encode(), password=None)
        assert isinstance(private_key, rsa.RSAPrivateKey)
        assert private_key.key_size == 4096
        line = private_key.public_key().public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH)
        assert result["public_key"] == f"{line.decode()} ci@example.org"
        blob = base64.b64decode(line.split()[1])
        assert result["fingerprint"] == "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")
        assert result["kind"] == "ssh-rsa"

    @pytest.mark.asyncio
    async def test_unknown_kind_lists_the_kinds_and_stores_nothing(self, env, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)

        with pytest.raises(ToolError, match=r"Unsupported key kind 'dsa'.*ssh-ed25519.*ssh-rsa"):
            await _fn(env, "generate_secret")(name="k", kind="dsa", project="p", domain="d")
        assert recorder.calls == []

    @pytest.mark.asyncio
    async def test_comment_must_be_a_single_line(self, env, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)

        with pytest.raises(ToolError, match="single line"):
            await _fn(env, "generate_secret")(name="k", comment="a\nssh-rsa AAAA", project="p", domain="d")
        assert recorder.calls == []

    @pytest.mark.asyncio
    async def test_empty_comment_leaves_no_trailing_space(self, env, monkeypatch):
        monkeypatch.setattr(flyte.remote.Secret, "create", _Recorder())

        result = await _fn(env, "generate_secret")(name="k", comment="", project="p", domain="d")

        assert result["public_key"].startswith("ssh-ed25519 ")
        assert result["public_key"] == result["public_key"].strip()
        assert len(result["public_key"].split(" ")) == 2

    @pytest.mark.asyncio
    async def test_existing_secret_is_refused_before_anything_is_generated_or_written(
        self, env, monkeypatch, caplog, secret_get
    ):
        secret_get.exists = True
        create = _Recorder()
        update = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", create)
        monkeypatch.setattr(flyte.remote.Secret, "update", update)

        with pytest.raises(ToolError, match=r"already exists.*nothing was changed.*another name.*delete_secret"):
            await _fn(env, "generate_secret")(name="deploy-key", project="p", domain="d")

        assert secret_get.calls == [{"args": ("deploy-key",), "scope": ("p", "d")}]
        assert create.calls == [] and update.calls == []

    @pytest.mark.asyncio
    async def test_a_create_error_after_the_check_is_a_sentence_and_nothing_is_overwritten(
        self, env, monkeypatch, caplog
    ):
        # The secret appeared between the existence check and the write (check-then-write race).
        calls = []

        async def create(**kwargs):
            calls.append(kwargs)
            raise ConnectError(Code.ALREADY_EXISTS, "secret deploy-key already exists")

        monkeypatch.setattr(flyte.remote.Secret, "create", SimpleNamespace(aio=create))
        update = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "update", update)

        with pytest.raises(ToolError, match=r"Could not create secret 'deploy-key'.*already exists") as exc_info:
            await _fn(env, "generate_secret")(name="deploy-key", project="p", domain="d")

        assert len(calls) == 1 and update.calls == []
        assert "BEGIN OPENSSH" not in str(exc_info.value)
        assert "BEGIN OPENSSH" not in caplog.text

    @pytest.mark.asyncio
    async def test_an_error_that_quotes_the_key_is_redacted(self, env, monkeypatch):
        async def create(**kwargs):
            raise RuntimeError(f"rejected {kwargs['value']}")

        monkeypatch.setattr(flyte.remote.Secret, "create", SimpleNamespace(aio=create))

        with pytest.raises(ToolError, match=r"RuntimeError: rejected \[redacted\]") as exc_info:
            await _fn(env, "generate_secret")(name="k", project="p", domain="d")
        assert "OPENSSH" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_cryptography_names_the_package(self, env, monkeypatch):
        real_import = builtins.__import__

        def no_cryptography(name, *args, **kwargs):
            if name.startswith("cryptography"):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_cryptography)

        with pytest.raises(ToolError, match="cryptography"):
            await _fn(env, "generate_secret")(name="k", project="p", domain="d")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kwargs", [{}, {"allow_inline_secret_values": True}, {"transport": "stdio"}])
    async def test_available_with_and_without_inline_values_and_over_stdio(self, init_config, monkeypatch, kwargs):
        monkeypatch.setattr(flyte.remote.Secret, "create", _Recorder())
        env = FlyteMCPAppEnvironment(name="test-mcp", **kwargs)

        result = await _fn(env, "generate_secret")(name="k", project="p", domain="d")

        assert result["created"] is True

    def test_schema_takes_a_name_and_no_value(self):
        schema = FlyteMCPAppEnvironment(name="test-mcp")._mcp_server._tool_manager._tools["generate_secret"].parameters
        assert schema["required"] == ["name"]
        assert set(schema["properties"]) == {"name", "kind", "comment", "project", "domain"}


# ------------------------------
# Ingress validators
# ------------------------------


async def _token_check(value: str) -> str | None:
    return None if value == VALUE else "The provider rejected this token."


class TestValidators:
    @pytest.fixture
    def env(self, init_config):
        return FlyteMCPAppEnvironment(name="test-mcp", secret_validators={"api-token": _token_check})

    @pytest.fixture
    def client(self, env):
        return TestClient(env._starlette_app, headers=AUTH, raise_server_exceptions=False)

    @pytest.mark.parametrize("inline", [False, True])
    @pytest.mark.parametrize("tool", ["create_secret", "update_secret"])
    def test_validator_is_absent_from_the_schema_when_none_are_registered(self, tool, inline):
        env = FlyteMCPAppEnvironment(name="test-mcp", allow_inline_secret_values=inline)
        assert "validator" not in env._mcp_server._tool_manager._tools[tool].parameters["properties"]

    def test_an_empty_registry_is_the_same_as_none(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", secret_validators={})
        assert "validator" not in env._mcp_server._tool_manager._tools["create_secret"].parameters["properties"]

    @pytest.mark.parametrize("inline", [False, True])
    @pytest.mark.parametrize("tool", ["create_secret", "update_secret"])
    def test_validator_is_optional_and_named_in_the_description_when_registered(self, tool, inline):
        env = FlyteMCPAppEnvironment(
            name="test-mcp", allow_inline_secret_values=inline, secret_validators={"api-token": _token_check}
        )
        registered = env._mcp_server._tool_manager._tools[tool]
        assert "validator" in registered.parameters["properties"]
        assert "validator" not in registered.parameters["required"]
        assert "api-token" in registered.description

    def test_registered_names_show_in_rich_repr(self, env):
        assert dict(env.__rich_repr__())["secret_validators"] == ["api-token"]
        assert "secret_validators" not in dict(FlyteMCPAppEnvironment(name="t").__rich_repr__())

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool", ["create_secret", "update_secret"])
    async def test_unknown_name_lists_the_registered_ones(self, env, tool):
        with pytest.raises(ToolError, match=r"Unknown validator 'nope'.*\['api-token'\]"):
            await _fn(env, tool)(name="s", validator="nope", project="p", domain="d", ctx=_ctx())
        assert env._secret_inputs._pending == {}

    @pytest.mark.asyncio
    async def test_a_name_with_none_registered_is_unknown_too(self, init_config):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        with pytest.raises(ToolError, match=r"Unknown validator 'x'.*\[\]"):
            await _fn(env, "create_secret")(name="s", validator="x", project="p", domain="d", ctx=_ctx())

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool", ["create_secret", "update_secret"])
    async def test_the_name_is_stored_on_the_pending_token(self, env, secret_get, tool):
        secret_get.exists = tool == "update_secret"
        result = await _fn(env, tool)(name="s", validator="api-token", project="p", domain="d", ctx=_ctx())
        assert env._secret_inputs.get(_token_of(result["url"])).validator == "api-token"

    @pytest.mark.asyncio
    async def test_cli_fallback_says_it_does_not_validate(self, env):
        result = await _fn(env, "create_secret")(name="s", validator="api-token", project="p", domain="d", ctx=None)
        assert "does not run the validator" in result["message"]

    def test_rejected_value_leaves_the_token_usable_and_skips_the_sdk(self, env, client, monkeypatch, caplog):
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)
        token, pending = env._secret_inputs.mint("create", "s", "p", "d", validator="api-token")

        bad = client.post(f"/secret-input/{token}", data={"value": "typo"})

        assert bad.status_code == 422
        assert "The provider rejected this token." in bad.text
        assert "still valid" in bad.text
        assert pending.used is False and recorder.calls == []

        good = client.post(f"/secret-input/{token}", data={"value": VALUE})

        assert good.status_code == 200
        assert recorder.calls == [{"name": "s", "value": VALUE, "scope": ("p", "d")}]
        assert VALUE not in good.text and VALUE not in caplog.text

    def test_a_message_that_quotes_the_value_is_redacted(self, init_config, monkeypatch):
        async def echo(value: str) -> str | None:
            return f"{value} is not a token"

        env = FlyteMCPAppEnvironment(name="test-mcp", secret_validators={"echo": echo})
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)
        token, _ = env._secret_inputs.mint("create", "s", "p", "d", validator="echo")

        response = TestClient(env._starlette_app, headers=AUTH).post(f"/secret-input/{token}", data={"value": VALUE})

        assert response.status_code == 422
        assert VALUE not in response.text and "[redacted] is not a token" in response.text

    def test_a_validator_that_raises_is_a_rejection_naming_only_the_class(self, init_config, monkeypatch, caplog):
        async def boom(value: str) -> str | None:
            raise ConnectionError(f"could not reach host with {value}")

        env = FlyteMCPAppEnvironment(name="test-mcp", secret_validators={"boom": boom})
        client = TestClient(env._starlette_app, headers=AUTH, raise_server_exceptions=False)
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)
        token, pending = env._secret_inputs.mint("create", "s", "p", "d", validator="boom")

        response = client.post(f"/secret-input/{token}", data={"value": VALUE})

        assert response.status_code == 422
        assert "ConnectionError" in response.text
        assert VALUE not in response.text and "could not reach" not in response.text
        assert VALUE not in caplog.text and "could not reach" not in caplog.text
        assert "ConnectionError" in caplog.text
        assert pending.used is False and recorder.calls == []

    def test_a_validator_that_times_out_is_a_rejection(self, init_config, monkeypatch):
        async def slow(value: str) -> str | None:
            await asyncio.sleep(5)
            return None

        monkeypatch.setattr(_secret_input, "VALIDATOR_TIMEOUT_S", 0.01)
        env = FlyteMCPAppEnvironment(name="test-mcp", secret_validators={"slow": slow})
        client = TestClient(env._starlette_app, headers=AUTH, raise_server_exceptions=False)
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)
        token, pending = env._secret_inputs.mint("create", "s", "p", "d", validator="slow")

        response = client.post(f"/secret-input/{token}", data={"value": VALUE})

        assert response.status_code == 422
        assert "did not answer" in response.text
        assert pending.used is False and recorder.calls == []

    def test_the_timeout_is_fifteen_seconds(self):
        assert _secret_input.VALIDATOR_TIMEOUT_S == 15

    def test_a_token_without_a_validator_skips_validation(self, env, client, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)
        token, _ = env._secret_inputs.mint("create", "s", "p", "d")

        assert client.post(f"/secret-input/{token}", data={"value": "anything"}).status_code == 200
        assert len(recorder.calls) == 1


class TestInlineValidators:
    @pytest.fixture
    def env(self, init_config):
        return FlyteMCPAppEnvironment(
            name="test-mcp", allow_inline_secret_values=True, secret_validators={"api-token": _token_check}
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("op", "tool"), [("create", "create_secret"), ("update", "update_secret")])
    async def test_rejected_value_is_a_tool_error_and_the_sdk_is_not_called(self, env, monkeypatch, caplog, op, tool):
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, op, recorder)

        with pytest.raises(ToolError, match="The provider rejected this token") as exc_info:
            await _fn(env, tool)(name="s", value="typo", validator="api-token", project="p", domain="d")

        assert recorder.calls == []
        assert "typo" not in caplog.text and "typo" not in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("op", "tool"), [("create", "create_secret"), ("update", "update_secret")])
    async def test_valid_value_is_saved(self, env, monkeypatch, secret_get, op, tool):
        secret_get.exists = op == "update"
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, op, recorder)

        await _fn(env, tool)(name="s", value=VALUE, validator="api-token", project="p", domain="d")

        assert [c["value"] for c in recorder.calls] == [VALUE]

    @pytest.mark.asyncio
    async def test_unknown_name_is_refused_before_the_sdk(self, env, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "update", recorder)

        with pytest.raises(ToolError, match="Unknown validator 'nope'"):
            await _fn(env, "update_secret")(name="s", value=VALUE, validator="nope", project="p", domain="d")
        assert recorder.calls == []

    @pytest.mark.asyncio
    async def test_no_validator_means_no_check(self, env, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)

        await _fn(env, "create_secret")(name="s", value="anything", project="p", domain="d")

        assert len(recorder.calls) == 1


# ------------------------------
# Edges of the form plumbing
# ------------------------------


class TestExistenceChecks:
    """The control plane replaces on create and creates on update, so the server checks first."""

    @pytest.fixture
    def env(self, init_config):
        return FlyteMCPAppEnvironment(name="test-mcp")

    @pytest.fixture
    def inline_env(self, init_config):
        return FlyteMCPAppEnvironment(name="test-mcp", allow_inline_secret_values=True)

    @pytest.fixture
    def client(self, env):
        return TestClient(env._starlette_app, headers=AUTH, raise_server_exceptions=False)

    @pytest.fixture
    def writers(self, monkeypatch):
        create, update = _Recorder(), _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", create)
        monkeypatch.setattr(flyte.remote.Secret, "update", update)
        return create, update

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ctx", [_ctx(), _ctx(url_elicitation=True)], ids=["result", "elicitation"])
    async def test_create_on_an_existing_secret_is_refused_before_a_token_is_minted(
        self, env, secret_get, writers, ctx
    ):
        secret_get.exists = True

        with pytest.raises(ToolError, match=r"'s' already exists.*nothing was changed.*update_secret"):
            await _fn(env, "create_secret")(name="s", project="p", domain="d", ctx=ctx)

        assert env._secret_inputs._pending == {}
        assert secret_get.calls == [{"args": ("s",), "scope": ("p", "d")}]
        assert [w.calls for w in writers] == [[], []]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ctx", [_ctx(), _ctx(url_elicitation=True)], ids=["result", "elicitation"])
    async def test_update_on_a_missing_secret_is_refused_before_a_token_is_minted(self, env, secret_get, writers, ctx):
        with pytest.raises(ToolError, match=r"'s' does not exist.*nothing was changed.*create_secret"):
            await _fn(env, "update_secret")(name="s", project="p", domain="d", ctx=ctx)

        assert env._secret_inputs._pending == {}
        assert secret_get.calls == [{"args": ("s",), "scope": ("p", "d")}]
        assert [w.calls for w in writers] == [[], []]

    @pytest.mark.asyncio
    async def test_stdio_checks_before_returning_the_fallback(self, init_config, secret_get, writers):
        env = FlyteMCPAppEnvironment(name="test-mcp", transport="stdio")

        secret_get.exists = True
        with pytest.raises(ToolError, match="already exists"):
            await _fn(env, "create_secret")(name="s", project="p", domain="d")
        secret_get.exists = False
        with pytest.raises(ToolError, match="does not exist"):
            await _fn(env, "update_secret")(name="s", project="p", domain="d")

        assert env._secret_inputs._pending == {}

    @pytest.mark.asyncio
    async def test_no_request_context_checks_before_returning_the_fallback(self, env, secret_get):
        secret_get.exists = True
        with pytest.raises(ToolError, match="already exists"):
            await _fn(env, "create_secret")(name="s", project="p", domain="d", ctx=None)

    @pytest.mark.asyncio
    async def test_inline_create_on_an_existing_secret_is_refused(self, inline_env, secret_get, writers):
        secret_get.exists = True

        with pytest.raises(ToolError, match=r"already exists.*nothing was changed.*update_secret") as exc_info:
            await _fn(inline_env, "create_secret")(name="s", value=VALUE, project="p", domain="d")

        assert [w.calls for w in writers] == [[], []]
        assert VALUE not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_inline_update_on_a_missing_secret_is_refused(self, inline_env, writers):
        with pytest.raises(ToolError, match=r"does not exist.*nothing was changed.*create_secret") as exc_info:
            await _fn(inline_env, "update_secret")(name="s", value=VALUE, project="p", domain="d")

        assert [w.calls for w in writers] == [[], []]
        assert VALUE not in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool", ["create_secret", "update_secret", "generate_secret"])
    @pytest.mark.parametrize("inline", [False, True])
    async def test_other_get_errors_propagate_and_nothing_is_written(
        self, init_config, secret_get, writers, tool, inline
    ):
        secret_get.exc = ConnectError(Code.PERMISSION_DENIED, "no")
        env = FlyteMCPAppEnvironment(name="test-mcp", allow_inline_secret_values=inline)
        kwargs = {"value": VALUE} if inline and tool != "generate_secret" else {}

        with pytest.raises(ConnectError) as exc_info:
            await _fn(env, tool)(name="s", project="p", domain="d", ctx=_ctx(), **kwargs)

        assert exc_info.value.code == Code.PERMISSION_DENIED
        assert env._secret_inputs._pending == {}
        assert [w.calls for w in writers] == [[], []]

    @pytest.mark.asyncio
    async def test_a_non_connect_error_also_propagates(self, env, secret_get):
        secret_get.exc = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            await _fn(env, "create_secret")(name="s", project="p", domain="d", ctx=_ctx())

    # The state can change during the 10 minutes the link is open.

    def test_post_on_a_secret_created_in_the_meantime_is_a_409_and_nothing_is_written(
        self, env, client, secret_get, writers
    ):
        token, pending = env._secret_inputs.mint("create", "s", "p", "d")
        secret_get.exists = True

        response = client.post(f"/secret-input/{token}", data={"value": VALUE})

        assert response.status_code == 409
        assert "already exists" in response.text and "nothing was changed" in response.text
        assert "ask the assistant for a new link" in response.text.lower()
        assert VALUE not in response.text
        assert [w.calls for w in writers] == [[], []]
        assert secret_get.calls == [{"args": ("s",), "scope": ("p", "d")}]
        assert pending.used is False

    def test_post_on_a_secret_deleted_in_the_meantime_is_a_409_and_nothing_is_written(self, env, client, writers):
        token, pending = env._secret_inputs.mint("update", "s", "p", "d")

        response = client.post(f"/secret-input/{token}", data={"value": VALUE})

        assert response.status_code == 409
        assert "does not exist" in response.text and "nothing was changed" in response.text
        assert "ask the assistant for a new link" in response.text.lower()
        assert VALUE not in response.text
        assert [w.calls for w in writers] == [[], []]
        assert pending.used is False

    def test_the_check_runs_after_validation(self, init_config, secret_get, writers):
        env = FlyteMCPAppEnvironment(name="test-mcp", secret_validators={"api-token": _token_check})
        client = TestClient(env._starlette_app, headers=AUTH, raise_server_exceptions=False)
        token, _ = env._secret_inputs.mint("create", "s", "p", "d", validator="api-token")

        assert client.post(f"/secret-input/{token}", data={"value": "bad"}).status_code == 422

        assert secret_get.calls == []

    def test_post_when_the_check_itself_fails_is_a_502_and_the_link_stays_usable(
        self, env, client, secret_get, writers, caplog
    ):
        token, pending = env._secret_inputs.mint("create", "s", "p", "d")
        secret_get.exc = RuntimeError(f"unreachable while checking {VALUE}")

        response = client.post(f"/secret-input/{token}", data={"value": VALUE})

        assert response.status_code == 502
        assert "RuntimeError" in response.text and "nothing was changed" in response.text.lower()
        assert VALUE not in response.text and VALUE not in caplog.text
        assert [w.calls for w in writers] == [[], []]
        assert pending.used is False

        secret_get.exc = None
        assert client.post(f"/secret-input/{token}", data={"value": VALUE}).status_code == 200
        assert [c["value"] for c in writers[0].calls] == [VALUE]


class TestEdges:
    def test_http_request_outside_a_request_is_none(self):
        class _NoRequest:
            @property
            def request_context(self):
                raise ValueError("Context is not available outside of a request")

        assert _secret_input._http_request(_NoRequest()) is None

    def test_request_not_routed_through_a_mount_uses_its_root_path(self):
        request = _ctx(prefix="/proxy", mount="", app_root_path=False).request_context.request
        assert _secret_input._public_base_url(request) == "http://testserver/proxy"

    def test_body_larger_than_the_limit_is_refused_even_if_content_length_lies(self, init_config, monkeypatch):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        recorder = _Recorder()
        monkeypatch.setattr(flyte.remote.Secret, "create", recorder)
        token, _ = env._secret_inputs.mint("create", "s", "p", "d")
        client = TestClient(env._starlette_app, headers=AUTH, raise_server_exceptions=False)

        response = client.post(
            f"/secret-input/{token}",
            content=b"value=" + b"x" * _secret_input.MAX_FORM_BYTES,
            headers={"Content-Type": "application/x-www-form-urlencoded", "Content-Length": "10"},
        )

        assert response.status_code == 413
        assert recorder.calls == []
