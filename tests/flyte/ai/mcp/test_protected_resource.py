"""RFC 9728 protected resource metadata on `FlyteMCPAppEnvironment`."""

from types import SimpleNamespace

import pytest

import flyte
from flyte.ai.mcp import FlyteMCPAppEnvironment, ProtectedResourceMetadata, _flyte_mcp_app

AUTH_SERVER = "https://auth.example.com"
SUFFIXED = "/.well-known/oauth-protected-resource/flyte-mcp/mcp"
ROOT = "/.well-known/oauth-protected-resource"

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "0"},
    },
}
MCP_HEADERS = {"Accept": "application/json, text/event-stream"}
MALFORMED_HOSTS = [
    'evil.example", scope="admin',
    "evil.example\\suffix",
    "a.example,scope=admin",
    "a.example;x=y",
    "a.example:",
    "a.example:99999",
    "-a.example",
    "a..example",
    "a_b.example",
    f"{'a' * 64}.example",
    ".".join(["a" * 63] * 3 + ["a" * 62]),
    "999.1.1.1",
]
# Challenge parameter names the reference MCP client would find inside the quoted URL.
CHALLENGE_NAME_RESOURCES = [
    "https://mcp.example.com/flyte-mcp/mcp?x=1&scope=admin",
    "https://mcp.example.com/flyte-mcp/mcp;scope=admin",
    "https://mcp.example.com/xscope=admin/mcp",
    "https://mcp.example.com/mcp?RESOURCE_METADATA=https://evil.example",
    "https://mcp.example.com/mcp?error=invalid_token",
    "https://mcp.example.com/mcp;Realm=admin",
]


def _env(**prm_kwargs) -> FlyteMCPAppEnvironment:
    prm_kwargs.setdefault("authorization_servers", [AUTH_SERVER])
    return FlyteMCPAppEnvironment(
        name="test-mcp",
        requires_auth=False,
        protected_resource_metadata=ProtectedResourceMetadata(**prm_kwargs),
    )


def _client(env: FlyteMCPAppEnvironment, **kwargs):
    from starlette.testclient import TestClient

    return TestClient(env._starlette_app, raise_server_exceptions=False, **kwargs)


@pytest.fixture
def fake_init_passthrough(monkeypatch):
    calls = []

    class FakeInitPassthrough:
        async def aio(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(flyte, "init_passthrough", FakeInitPassthrough())
    return calls


class TestConfig:
    def test_exported_from_package(self):
        import flyte.ai.mcp

        assert "ProtectedResourceMetadata" in flyte.ai.mcp.__all__

    def test_off_by_default(self):
        assert FlyteMCPAppEnvironment(name="test-mcp").protected_resource_metadata is None

    def test_empty_authorization_servers_rejected(self):
        with pytest.raises(ValueError, match="authorization_servers must not be empty"):
            ProtectedResourceMetadata(authorization_servers=[])

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"authorization_servers": ["auth.example.com"]},
            {"resource": "/flyte-mcp/mcp"},
            {"resource_documentation": "ftp://docs.example.com"},
            {"authorization_servers": ["https://auth.example,scope=admin"]},
            {"resource_documentation": 'https://docs.example.com/a"b'},
        ],
    )
    def test_non_http_urls_rejected(self, kwargs):
        with pytest.raises(ValueError, match="absolute http"):
            ProtectedResourceMetadata(**kwargs)

    def test_stdio_rejected(self):
        with pytest.raises(ValueError, match="requires an HTTP transport"):
            FlyteMCPAppEnvironment(
                name="test-mcp",
                transport="stdio",
                protected_resource_metadata=ProtectedResourceMetadata(authorization_servers=[AUTH_SERVER]),
            )

    def test_rich_repr_includes_metadata(self):
        env = _env()
        assert dict(env.__rich_repr__())["protected_resource_metadata"] == env.protected_resource_metadata


class TestDisabled:
    """Without the config, nothing about the HTTP surface changes."""

    def test_requires_auth_keeps_plain_bearer_challenge(self):
        env = FlyteMCPAppEnvironment(name="test-mcp")
        client = _client(env)
        response = client.post("/flyte-mcp/mcp", json={})
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        # The metadata path is just another authenticated path.
        assert client.get(ROOT).status_code == 401

    def test_requires_auth_installs_only_passthrough_middleware(self):
        from flyte.app.extras import FastAPIPassthroughAuthMiddleware

        env = FlyteMCPAppEnvironment(name="test-mcp")
        middleware = env._starlette_middleware()
        assert [m.cls for m in middleware] == [FastAPIPassthroughAuthMiddleware]
        assert middleware[0].kwargs["excluded_paths"] == {"/health"}

    def test_no_auth_serves_no_metadata_and_installs_no_middleware(self):
        env = FlyteMCPAppEnvironment(name="test-mcp", requires_auth=False)
        assert env._starlette_middleware() == []
        client = _client(env)
        assert client.get(SUFFIXED).status_code == 404
        assert client.get(ROOT).status_code == 404

    @pytest.mark.asyncio
    async def test_no_auth_skips_passthrough_init(self, fake_init_passthrough):
        await FlyteMCPAppEnvironment(name="test-mcp", requires_auth=False)._starlette_lifespan_startup()
        assert fake_init_passthrough == []


class TestMetadataDocument:
    @pytest.mark.parametrize("path", [SUFFIXED, ROOT])
    def test_served_without_auth(self, path):
        response = _client(_env()).get(path)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert response.json() == {
            "resource": "http://testserver/flyte-mcp/mcp",
            "authorization_servers": [AUTH_SERVER],
            "bearer_methods_supported": ["header"],
        }

    def test_optional_fields(self):
        env = _env(
            scopes_supported=["all"],
            resource_name="Flyte MCP",
            resource_documentation="https://docs.example.com/mcp",
        )
        body = _client(env).get(SUFFIXED).json()
        assert body["scopes_supported"] == ["all"]
        assert body["resource_name"] == "Flyte MCP"
        assert body["resource_documentation"] == "https://docs.example.com/mcp"

    def test_urls_are_advertised_verbatim(self):
        # A bare-host issuer must not gain a trailing slash: clients compare it with the
        # authorization server's own `issuer` as a string.
        env = _env(resource="https://mcp.example.com")
        body = _client(env).get(ROOT).json()
        assert body["authorization_servers"] == ["https://auth.example.com"]
        assert body["resource"] == "https://mcp.example.com"

    def test_cors_preflight_allowed(self):
        response = _client(_env()).options(
            SUFFIXED,
            headers={"Origin": "https://inspector.example.com", "Access-Control-Request-Method": "GET"},
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "*"

    def test_excluded_from_passthrough_middleware(self):
        from flyte.app.extras import FastAPIPassthroughAuthMiddleware

        middleware = _env()._starlette_middleware()
        passthrough = next(m for m in middleware if m.cls is FastAPIPassthroughAuthMiddleware)
        assert passthrough.kwargs["excluded_paths"] == {"/health", SUFFIXED, ROOT}

    def test_health_still_unauthenticated(self):
        assert _client(_env()).get("/health").status_code == 200


class TestResourceAndPaths:
    def test_resource_derived_from_request_host(self):
        client = _client(_env(), base_url="https://mcp.example.com")
        assert client.get(SUFFIXED).json()["resource"] == "https://mcp.example.com/flyte-mcp/mcp"

    def test_custom_mount_path(self):
        env = FlyteMCPAppEnvironment(
            name="test-mcp",
            requires_auth=False,
            mcp_mount_path="/tools",
            protected_resource_metadata=ProtectedResourceMetadata(authorization_servers=[AUTH_SERVER]),
        )
        response = _client(env).get("/.well-known/oauth-protected-resource/tools/mcp")
        assert response.status_code == 200
        assert response.json()["resource"] == "http://testserver/tools/mcp"

    def test_sse_transport_uses_sse_endpoint(self):
        env = FlyteMCPAppEnvironment(
            name="test-mcp",
            requires_auth=False,
            transport="sse",
            protected_resource_metadata=ProtectedResourceMetadata(authorization_servers=[AUTH_SERVER]),
        )
        response = _client(env).get("/.well-known/oauth-protected-resource/flyte-mcp/sse")
        assert response.status_code == 200
        assert response.json()["resource"] == "http://testserver/flyte-mcp/sse"

    def test_resource_override_is_used_and_moves_suffixed_path(self):
        # A proxy that serves the app under a prefix: the RFC places the metadata after the
        # host and before the resource's own path.
        env = _env(resource="https://public.example.com/team/flyte-mcp/mcp")
        client = _client(env)
        response = client.get("/.well-known/oauth-protected-resource/team/flyte-mcp/mcp")
        assert response.status_code == 200
        assert response.json()["resource"] == "https://public.example.com/team/flyte-mcp/mcp"
        assert client.get(ROOT).json()["resource"] == "https://public.example.com/team/flyte-mcp/mcp"

        challenge = client.post("/flyte-mcp/mcp", json={}).headers["www-authenticate"]
        assert challenge == (
            'Bearer resource_metadata="https://public.example.com'
            '/.well-known/oauth-protected-resource/team/flyte-mcp/mcp"'
        )


class TestChallenge:
    def test_missing_token_gets_rfc9728_challenge(self):
        response = _client(_env()).post("/flyte-mcp/mcp", json=INITIALIZE, headers=MCP_HEADERS)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == (
            'Bearer resource_metadata="http://testserver/.well-known/oauth-protected-resource/flyte-mcp/mcp"'
        )
        assert response.json() == {"detail": "Authentication credentials required"}

    def test_challenge_url_follows_request_host(self):
        client = _client(_env(), base_url="https://mcp.example.com")
        challenge = client.post("/flyte-mcp/mcp", json={}).headers["www-authenticate"]
        assert challenge == f'Bearer resource_metadata="https://mcp.example.com{SUFFIXED}"'

    def test_challenge_carries_scopes(self):
        challenge = _client(_env(scopes_supported=["read", "write"])).post("/flyte-mcp/mcp", json={})
        assert challenge.headers["www-authenticate"].endswith(', scope="read write"')

    @pytest.mark.parametrize(
        "headers",
        [
            {"Authorization": "Basic dXNlcjpwYXNz"},
            {"Authorization": "Bearer "},
            {"Cookie": "session=abc"},
        ],
    )
    def test_anything_but_a_bearer_token_is_challenged(self, headers):
        response = _client(_env()).post("/flyte-mcp/mcp", json={}, headers=headers)
        assert response.status_code == 401
        assert "resource_metadata=" in response.headers["www-authenticate"]

    @pytest.mark.parametrize("host", [*MALFORMED_HOSTS, "testserver"])
    def test_reference_client_reads_only_the_configured_scope(self, host):
        import httpx
        from mcp.client.auth.utils import extract_scope_from_www_auth

        response = _client(_env(scopes_supported=["read"])).post("/flyte-mcp/mcp", headers={"Host": host})
        scope = extract_scope_from_www_auth(httpx.Response(401, headers=response.headers))
        if host == "testserver":
            assert response.status_code == 401
            assert scope == "read"
        else:
            assert response.status_code == 400
            assert scope is None

    @pytest.mark.parametrize("resource", CHALLENGE_NAME_RESOURCES)
    def test_challenge_parameter_names_in_resource_are_rejected(self, resource):
        with pytest.raises(ValueError, match="WWW-Authenticate"):
            ProtectedResourceMetadata(resource=resource, scopes_supported=["read"])

    def test_challenge_parameter_names_in_mount_path_are_rejected(self):
        with pytest.raises(ValueError, match="WWW-Authenticate"):
            FlyteMCPAppEnvironment(
                name="test-mcp",
                requires_auth=False,
                mcp_mount_path="/x;scope=admin",
                protected_resource_metadata=ProtectedResourceMetadata(authorization_servers=[AUTH_SERVER]),
            )

    def test_unsafe_request_derived_resource_is_not_challenged(self):
        # Construction validates the mount path; the per-request check still guards a path
        # changed afterwards, answering 400 without a challenge.
        env = _env(scopes_supported=["read"])
        env.mcp_mount_path = "/x;scope=admin"
        response = _client(env).post("/flyte-mcp/mcp")
        assert response.status_code == 400
        assert "www-authenticate" not in response.headers

    def test_reference_client_reads_the_exact_challenge(self):
        import httpx
        from mcp.client.auth.utils import extract_resource_metadata_from_www_auth, extract_scope_from_www_auth

        env = _env(resource="https://mcp.example.com/flyte-mcp/mcp?x=1", scopes_supported=["read"])
        response = _client(env).post("/flyte-mcp/mcp")
        assert response.status_code == 401
        parsed = httpx.Response(401, headers=response.headers)
        assert extract_scope_from_www_auth(parsed) == "read"
        assert extract_resource_metadata_from_www_auth(parsed) == (
            "https://mcp.example.com/.well-known/oauth-protected-resource/flyte-mcp/mcp?x=1"
        )

    def test_bearer_token_reaches_mcp_server(self, fake_init_passthrough):
        with _client(_env()) as client:
            response = client.post(
                "/flyte-mcp/mcp",
                json=INITIALIZE,
                headers={**MCP_HEADERS, "Authorization": "Bearer abc"},
            )
        assert response.status_code == 200, response.text
        assert response.json()["result"]["serverInfo"]["name"] == "test-mcp"

    def test_bearer_token_is_forwarded_to_flyte(self, monkeypatch):
        # The passthrough middleware still runs behind the challenge, so tool calls see the token.
        from flyte.app.extras import FastAPIPassthroughAuthMiddleware

        seen = []
        original = FastAPIPassthroughAuthMiddleware.extract_authorization_header

        def spy(request):
            result = original(request)
            seen.append(result)
            return result

        monkeypatch.setattr(FastAPIPassthroughAuthMiddleware, "extract_authorization_header", staticmethod(spy))
        _client(_env()).post("/flyte-mcp/mcp", json={}, headers={"Authorization": "Bearer abc"})
        assert seen == [("authorization", "Bearer abc")]


class TestPassthroughWithoutIngressAuth:
    """`requires_auth=False` with metadata still acts as the caller, never as the app."""

    def test_passthrough_middleware_installed(self):
        from flyte.ai.mcp._protected_resource import BearerChallengeMiddleware
        from flyte.app.extras import FastAPIPassthroughAuthMiddleware

        classes = [m.cls for m in _env()._starlette_middleware()]
        assert classes == [BearerChallengeMiddleware, FastAPIPassthroughAuthMiddleware]

    @pytest.mark.asyncio
    async def test_lifespan_initializes_passthrough(self, fake_init_passthrough):
        await _env()._starlette_lifespan_startup()
        assert len(fake_init_passthrough) == 1


class TestAuthorizationServerDiscovery:
    def test_discovered_issuer_is_advertised_and_cached(self, monkeypatch):
        calls = []

        async def fake_discover():
            calls.append(1)
            return "https://issuer.example.com/oauth2/default"

        monkeypatch.setattr(_flyte_mcp_app, "discover_authorization_server", fake_discover)
        client = _client(_env(authorization_servers=None))
        for _ in range(2):
            body = client.get(SUFFIXED).json()
            assert body["authorization_servers"] == ["https://issuer.example.com/oauth2/default"]
        assert len(calls) == 1

    def test_failed_discovery_returns_503_and_retries(self, monkeypatch):
        from connectrpc.code import Code
        from connectrpc.errors import ConnectError

        outcomes = [ConnectError(Code.UNAVAILABLE, "control plane down"), "https://issuer.example.com"]

        async def flaky_discover():
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(_flyte_mcp_app, "discover_authorization_server", flaky_discover)
        client = _client(_env(authorization_servers=None))
        failed = client.get(SUFFIXED)
        assert failed.status_code == 503
        assert "cache-control" not in failed.headers
        assert client.get(SUFFIXED).json()["authorization_servers"] == ["https://issuer.example.com"]

    @pytest.mark.asyncio
    async def test_discover_reads_issuer_from_auth_metadata_service(self, monkeypatch):
        import flyteidl2.auth.auth_service_connect as auth_connect

        import flyte._initialize
        from flyte.ai.mcp._protected_resource import discover_authorization_server

        constructed = {}

        class FakeAuthMetadataServiceClient:
            def __init__(self, address, http_client):
                constructed.update(address=address, http_client=http_client)

            async def get_o_auth2_metadata(self, request):
                return SimpleNamespace(issuer="https://issuer.example.com")

        http_client = object()
        session = SimpleNamespace(endpoint="https://flyte.example.com", http_client=http_client)
        cfg = SimpleNamespace(client=SimpleNamespace(session_config=session))
        monkeypatch.setattr(flyte._initialize, "_get_init_config", lambda: cfg)
        monkeypatch.setattr(auth_connect, "AuthMetadataServiceClient", FakeAuthMetadataServiceClient)

        assert await discover_authorization_server() == "https://issuer.example.com"
        # The session's http client, without the passthrough interceptors.
        assert constructed == {"address": "https://flyte.example.com", "http_client": http_client}

    @pytest.mark.asyncio
    async def test_discover_requires_initialized_sdk(self, monkeypatch):
        import flyte._initialize
        from flyte.ai.mcp._protected_resource import discover_authorization_server

        monkeypatch.setattr(flyte._initialize, "_get_init_config", lambda: None)
        with pytest.raises(RuntimeError, match="not initialized"):
            await discover_authorization_server()

    @pytest.mark.asyncio
    async def test_discover_rejects_missing_issuer(self, monkeypatch):
        import flyteidl2.auth.auth_service_connect as auth_connect

        import flyte._initialize
        from flyte.ai.mcp._protected_resource import discover_authorization_server

        class FakeAuthMetadataServiceClient:
            def __init__(self, address, http_client):
                pass

            async def get_o_auth2_metadata(self, request):
                return SimpleNamespace(issuer="")

        session = SimpleNamespace(endpoint="https://flyte.example.com", http_client=None)
        cfg = SimpleNamespace(client=SimpleNamespace(session_config=session))
        monkeypatch.setattr(flyte._initialize, "_get_init_config", lambda: cfg)
        monkeypatch.setattr(auth_connect, "AuthMetadataServiceClient", FakeAuthMetadataServiceClient)

        with pytest.raises(RuntimeError, match="advertises no OAuth issuer"):
            await discover_authorization_server()


class TestResourceHardening:
    def test_explicit_resource_overrides_untrusted_request(self, monkeypatch):
        monkeypatch.setenv("INTERNAL_APP_ENDPOINT_PATTERN", "https://{app_fqdn}.apps.example.com/team/")
        client = _client(
            _env(resource="https://test-mcp.apps.example.com/team/flyte-mcp/mcp"),
            base_url="http://internal:8080",
        )
        path = "/.well-known/oauth-protected-resource/team/flyte-mcp/mcp"
        response = client.get(path, headers={"Host": "attacker.example", "X-Forwarded-Proto": "http"})
        assert response.status_code == 200
        assert response.json()["resource"] == "https://test-mcp.apps.example.com/team/flyte-mcp/mcp"
        challenge = client.post("/flyte-mcp/mcp").headers["www-authenticate"]
        assert challenge == f'Bearer resource_metadata="https://test-mcp.apps.example.com{path}"'

    def test_explicit_resource_wins_over_ingress_endpoint_pattern(self, monkeypatch):
        monkeypatch.setenv("INTERNAL_APP_ENDPOINT_PATTERN", "https://{app_fqdn}.apps.example.com")
        response = _client(_env(resource="https://custom.example.com/mcp")).get(ROOT)
        assert response.json()["resource"] == "https://custom.example.com/mcp"

    def test_request_dependent_metadata_is_not_shared_cacheable(self):
        client = _client(_env())
        for host in ["first.example.com", "second.example.com"]:
            response = client.get(ROOT, headers={"Host": host})
            assert response.json()["resource"] == f"http://{host}/flyte-mcp/mcp"
            assert response.headers["cache-control"] == "no-store"

    def test_raw_forwarding_headers_are_not_trusted(self):
        response = _client(_env()).get(
            ROOT,
            headers={
                "X-Forwarded-Host": "attacker.example.com",
                "X-Forwarded-Proto": "https",
                "Forwarded": "host=attacker.example.com;proto=https",
            },
        )
        assert response.json()["resource"] == "http://testserver/flyte-mcp/mcp"

    @pytest.mark.parametrize("host", MALFORMED_HOSTS)
    @pytest.mark.parametrize("path", [ROOT, "/flyte-mcp/mcp"])
    def test_malformed_request_authority_is_rejected(self, host, path):
        response = _client(_env()).get(path, headers={"Host": host})
        assert response.status_code == 400
        assert "www-authenticate" not in response.headers

    @pytest.mark.parametrize(
        "host",
        [
            "signin.hosted.unionai.cloud",
            "icy-voice-1db2f.apps.demo.hosted.unionai.cloud",
            "localhost:8080",
            "testserver",
            "[::1]:8080",
            "10.0.0.1:8080",
            f"{'a' * 63}.example",
            ".".join(["a" * 63] * 3 + ["a" * 61]),
        ],
    )
    def test_real_request_authorities_are_advertised(self, host):
        client = _client(_env())
        assert client.get(ROOT, headers={"Host": host}).json()["resource"] == f"http://{host}/flyte-mcp/mcp"
        challenge = client.post("/flyte-mcp/mcp", headers={"Host": host}).headers["www-authenticate"]
        assert challenge == f'Bearer resource_metadata="http://{host}{SUFFIXED}"'

    @pytest.mark.parametrize("path", [ROOT, "/flyte-mcp/mcp"])
    def test_repeated_host_headers_are_rejected(self, path):
        import httpx

        headers = httpx.Headers([("host", "a.example"), ("host", "b.example")])
        response = _client(_env()).get(path, headers=headers)
        assert response.status_code == 400
        assert "www-authenticate" not in response.headers

    @pytest.mark.parametrize("path", [ROOT, "/flyte-mcp/mcp"])
    def test_empty_port_does_not_fall_back_to_socket_address(self, path):
        response = _client(_env()).get(path, headers={"Host": "a.example:"})
        assert response.status_code == 400
        assert "testserver" not in response.text
        assert not any("testserver" in value for value in response.headers.values())

    @pytest.mark.parametrize("scope", ['read", error="injected', "read\r\nInjected: yes", "read write", "réad", ""])
    def test_invalid_scope_tokens_are_rejected(self, scope):
        with pytest.raises(ValueError, match="scope"):
            ProtectedResourceMetadata(scopes_supported=[scope])

    @pytest.mark.parametrize(
        "url",
        [
            'https://example.com/a"b',
            "https://example.com/a\\b",
            "https://example.com/\r\nb",
            "https://user:password@example.com/mcp",
            "https://example.com:invalid/mcp",
            "https://example.com/mcp#fragment",
            "https://a.example,scope=admin/mcp",
            "https://a.example;x=y/mcp",
            "https://a.example:/mcp",
            "https://a.example/mcp?x=1,scope=admin",
            "https://a_b.example/mcp",
            f"https://{'a' * 64}.example/mcp",
            f"https://{'.'.join(['a' * 63] * 3 + ['a' * 62])}/mcp",
            "https://999.1.1.1/mcp",
        ],
    )
    def test_unsafe_resource_urls_are_rejected(self, url):
        with pytest.raises(ValueError):
            ProtectedResourceMetadata(resource=url)

    @pytest.mark.parametrize("host", ["10.0.0.1:8080", f"{'a' * 63}.example", "[::1]:8080"])
    def test_configured_resource_hosts_are_accepted(self, host):
        resource = f"http://{host}/flyte-mcp/mcp"
        assert _client(_env(resource=resource)).get(ROOT).json()["resource"] == resource

    @pytest.mark.parametrize("mount_path", ["/x,scope=admin", "/x;scope=admin", "/x{id}"])
    def test_unsafe_mount_path_is_rejected_at_construction(self, mount_path):
        with pytest.raises(ValueError, match="mcp_mount_path"):
            FlyteMCPAppEnvironment(
                name="test-mcp",
                requires_auth=False,
                mcp_mount_path=mount_path,
                protected_resource_metadata=ProtectedResourceMetadata(authorization_servers=[AUTH_SERVER]),
            )
        # Without protected-resource metadata the path is never advertised, so it is unchanged.
        assert FlyteMCPAppEnvironment(name="test-mcp", mcp_mount_path=mount_path).mcp_mount_path == mount_path

    def test_fixed_resource_metadata_remains_cacheable(self):
        response = _client(_env(resource="https://public.example.com/mcp")).get(ROOT)
        assert response.headers["cache-control"] == "public, max-age=3600"

    def test_invalid_discovered_issuer_returns_503_and_retries(self, monkeypatch):
        outcomes = iter(["not-an-issuer", "https://issuer.example.com"])

        async def discover():
            return next(outcomes)

        monkeypatch.setattr(_flyte_mcp_app, "discover_authorization_server", discover)
        client = _client(_env(authorization_servers=None))
        assert client.get(ROOT).status_code == 503
        assert client.get(ROOT).json()["authorization_servers"] == ["https://issuer.example.com"]

    def test_scope_list_mutation_cannot_inject_challenge(self):
        scopes = ["read"]
        client = _client(_env(scopes_supported=scopes))
        scopes.append('write", error="injected')
        challenge = client.post("/flyte-mcp/mcp").headers["www-authenticate"]
        assert challenge.endswith(', scope="read"')

    def test_resource_path_parameters_are_preserved(self):
        resource = "https://public.example.com/team;v=1/mcp;v=2"
        client = _client(_env(resource=resource))
        path = "/.well-known/oauth-protected-resource/team;v=1/mcp;v=2"
        assert client.get(path).json()["resource"] == resource
        assert client.post("/flyte-mcp/mcp").headers["www-authenticate"] == (
            f'Bearer resource_metadata="https://public.example.com{path}"'
        )


class TestDiscoveryFailureBoundaries:
    @pytest.mark.parametrize("error_type", [RuntimeError, ValueError, TypeError, AttributeError])
    def test_unexpected_discovery_errors_propagate(self, monkeypatch, error_type):
        from starlette.testclient import TestClient

        error = error_type("discovery bug")

        async def broken_discover():
            raise error

        monkeypatch.setattr(_flyte_mcp_app, "discover_authorization_server", broken_discover)
        env = _env(authorization_servers=None)
        # Re-raised server exceptions show the error escaped the metadata endpoint unchanged.
        client = TestClient(env._starlette_app, base_url="https://mcp.example.com")
        with pytest.raises(error_type, match="discovery bug") as raised:
            client.get(SUFFIXED)
        assert raised.value is error
        assert env._discovered_authorization_server is None

    @pytest.mark.parametrize("failure", ["connection", "timeout", "url", "rpc"])
    def test_transport_failure_returns_503_and_retries(self, monkeypatch, failure):
        import httpx
        from connectrpc.code import Code
        from connectrpc.errors import ConnectError
        from flyteidl2.auth.auth_service_pb2 import GetOAuth2MetadataResponse

        import flyte._initialize

        errors = {
            "connection": httpx.ConnectError("connection refused"),
            "timeout": TimeoutError("control plane timed out"),
            "url": httpx.InvalidURL("invalid endpoint"),
            "rpc": ConnectError(Code.UNAVAILABLE, "control plane unavailable"),
        }
        calls = []

        class FlakyHTTPClient:
            async def get(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    raise errors[failure]
                return SimpleNamespace(
                    status=200,
                    headers={"content-type": "application/proto"},
                    content=GetOAuth2MetadataResponse(issuer="https://issuer.example.com").SerializeToString(),
                )

            post = get

        session = SimpleNamespace(endpoint="https://flyte.example.com", http_client=FlakyHTTPClient())
        cfg = SimpleNamespace(client=SimpleNamespace(session_config=session))
        monkeypatch.setattr(flyte._initialize, "_get_init_config", lambda: cfg)
        env = _env(authorization_servers=None)
        client = _client(env)
        failed = client.get(SUFFIXED)
        assert failed.status_code == 503
        assert "cache-control" not in failed.headers
        assert env._discovered_authorization_server is None
        for _ in range(2):
            assert client.get(SUFFIXED).json()["authorization_servers"] == ["https://issuer.example.com"]
        assert len(calls) == 2

    @pytest.mark.parametrize("missing_issuer", [False, True])
    def test_expected_configuration_failure_returns_503_and_retries(self, monkeypatch, missing_issuer):
        import flyteidl2.auth.auth_service_connect as auth_connect

        import flyte._initialize

        class FakeAuthMetadataServiceClient:
            def __init__(self, address, http_client):
                pass

            async def get_o_auth2_metadata(self, request):
                return SimpleNamespace(issuer="")

        session = SimpleNamespace(endpoint="https://flyte.example.com", http_client=None)
        cfg = SimpleNamespace(client=SimpleNamespace(session_config=session)) if missing_issuer else None
        monkeypatch.setattr(flyte._initialize, "_get_init_config", lambda: cfg)
        monkeypatch.setattr(auth_connect, "AuthMetadataServiceClient", FakeAuthMetadataServiceClient)
        env = _env(authorization_servers=None)
        client = _client(env)
        failed = client.get(SUFFIXED)
        assert failed.status_code == 503
        assert "cache-control" not in failed.headers
        assert env._discovered_authorization_server is None

        async def recovered_discover():
            return "https://issuer.example.com"

        monkeypatch.setattr(_flyte_mcp_app, "discover_authorization_server", recovered_discover)
        assert client.get(SUFFIXED).json()["authorization_servers"] == ["https://issuer.example.com"]

    @pytest.mark.parametrize("missing_dependency", [False, True])
    def test_transport_security_import_only_handles_import_errors(self, monkeypatch, missing_dependency):
        import builtins

        env = _env()
        original_import = builtins.__import__
        error = ImportError("optional dependency absent") if missing_dependency else RuntimeError("import bug")

        def broken_import(name, *args, **kwargs):
            if name == "mcp.server.transport_security":
                raise error
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", broken_import)
        if missing_dependency:
            assert env._transport_security_settings() is None
        else:
            with pytest.raises(RuntimeError, match="import bug") as raised:
                env._transport_security_settings()
            assert raised.value is error


class TestDiscoveryEdgeCases:
    @pytest.mark.parametrize("segment", ["team%20one", "team%2Fone", "team%252Fone", "team;one"])
    @pytest.mark.parametrize("query", ["", "?x=y"])
    @pytest.mark.asyncio
    async def test_encoded_resource_metadata_is_public(self, segment, query):
        from httpx import ASGITransport, AsyncClient

        resource = f"https://public.example/{segment}/mcp{query}"
        # Starlette's TestClient unquotes HTTPX's already-decoded path a second time;
        # ASGITransport preserves the single decoding used by an ASGI server.
        async with AsyncClient(
            transport=ASGITransport(app=_env(resource=resource)._starlette_app), base_url="https://public.example"
        ) as client:
            response = await client.post("/flyte-mcp/mcp")
            challenge = response.headers["www-authenticate"]
            metadata_url = f"https://public.example{ROOT}/{segment}/mcp{query}"
            assert challenge == f'Bearer resource_metadata="{metadata_url}"'
            for url in (ROOT, metadata_url):
                response = await client.get(url)
                assert response.status_code == 200
                assert response.json()["resource"] == resource

    def test_resource_query_is_preserved_in_challenge(self):
        resource = "https://public.example/mcp?x=y"
        client = _client(_env(resource=resource))
        metadata_url = f"https://public.example{ROOT}/mcp?x=y"
        assert client.post("/flyte-mcp/mcp").headers["www-authenticate"] == (
            f'Bearer resource_metadata="{metadata_url}"'
        )
        assert client.get(metadata_url).json()["resource"] == resource

    def test_internal_endpoint_pattern_is_not_public_configuration(self, monkeypatch):
        monkeypatch.setenv("INTERNAL_APP_ENDPOINT_PATTERN", "https://{app_fqdn}.apps.example/flyte-mcp/mcp")
        response = _client(_env(), base_url="https://public.example").get(ROOT)
        assert response.status_code == 200
        assert response.json()["resource"] == "https://public.example/flyte-mcp/mcp"
        assert response.headers["cache-control"] == "no-store"

    @pytest.mark.parametrize("base_path", ["", "/team", "/team/"])
    def test_explicit_resource_with_base_path(self, base_path):
        resource = f"https://test-mcp.apps.example{base_path.rstrip('/')}/flyte-mcp/mcp"
        client = _client(_env(resource=resource))
        assert client.get(ROOT).json()["resource"] == resource
        challenge = client.post("/flyte-mcp/mcp").headers["www-authenticate"]
        metadata_url = challenge.split('"')[1]
        assert client.get(metadata_url).json()["resource"] == resource

    @pytest.mark.parametrize(
        "issuer",
        [
            "http://issuer.example",
            "https://issuer.example/?query=x",
            "https://issuer.example/#fragment",
            "https://user:password@issuer.example",
        ],
    )
    def test_invalid_configured_issuer_is_rejected(self, issuer):
        with pytest.raises(ValueError, match="authorization_servers"):
            ProtectedResourceMetadata(authorization_servers=[issuer])

    @pytest.mark.parametrize(
        "issuer",
        [
            "http://issuer.example",
            "https://issuer.example/?query=x",
            "https://issuer.example/#fragment",
            "https://user:password@issuer.example",
        ],
    )
    def test_invalid_discovered_issuer_is_not_cached(self, monkeypatch, issuer):
        outcomes = iter([issuer, AUTH_SERVER])

        async def discover():
            return next(outcomes)

        monkeypatch.setattr(_flyte_mcp_app, "discover_authorization_server", discover)
        client = _client(_env(authorization_servers=None))
        failed = client.get(ROOT)
        assert failed.status_code == 503
        assert "cache-control" not in failed.headers
        assert client.get(ROOT).json()["authorization_servers"] == [AUTH_SERVER]
        assert client.get(ROOT).json()["authorization_servers"] == [AUTH_SERVER]

    @pytest.mark.parametrize("servers", [["http://issuer.example"], []])
    def test_direct_issuer_list_mutation_is_revalidated(self, servers):
        env = _env()
        client = _client(env)
        assert client.get(ROOT).status_code == 200
        env.protected_resource_metadata.authorization_servers[:] = servers
        failed = client.get(ROOT)
        assert failed.status_code == 503
        assert "cache-control" not in failed.headers
        env.protected_resource_metadata.authorization_servers[:] = [AUTH_SERVER]
        assert client.get(ROOT).json()["authorization_servers"] == [AUTH_SERVER]

    def test_local_resource_can_use_http(self):
        resource = "http://localhost:8080/mcp"
        assert _client(_env(resource=resource)).get(ROOT).json()["resource"] == resource


class TestUnrepresentableResourcePaths:
    @pytest.mark.parametrize(
        "segment",
        [
            "team%3Fone",
            "team%23one",
            "team{id:int}",
            "team%7Bid%3Aint%7D",
            "team{id:bogus}",
            "team%7Bid%3Abogus%7D",
            "team{",
            "team%7B",
            "team}",
            "team%7D",
        ],
    )
    def test_unroutable_resource_path_is_rejected(self, segment):
        with pytest.raises(ValueError, match=r"resource.*path cannot be routed"):
            ProtectedResourceMetadata(resource=f"https://public.example/{segment}/mcp?x=y")

    @pytest.mark.parametrize("segment", ["team%20one", "team%2Fone", "team%252Fone", "team;one"])
    @pytest.mark.asyncio
    async def test_routable_explicit_resource_path_remains_public(self, segment):
        from httpx import ASGITransport, AsyncClient

        resource = f"http://localhost:8080/{segment}/flyte-mcp/mcp"
        async with AsyncClient(
            transport=ASGITransport(app=_env(resource=resource)._starlette_app), base_url="http://localhost:8080"
        ) as client:
            response = await client.post("/flyte-mcp/mcp")
            metadata_url = response.headers["www-authenticate"].split('"')[1]
            for url in (ROOT, metadata_url):
                response = await client.get(url)
                assert response.status_code == 200
                assert response.json()["resource"] == resource

    def test_path_limitation_is_specific_to_resource_urls(self):
        issuer = "https://auth.example/team%3Fone/{id}"
        documentation = "https://docs.example/team%23one/{id}?x=y"
        prm = ProtectedResourceMetadata(authorization_servers=[issuer], resource_documentation=documentation)
        assert prm.authorization_servers == [issuer]
        assert prm.resource_documentation == documentation


class TestLiveIngressResource:
    def test_internal_service_pattern_does_not_override_public_origin(self, monkeypatch):
        # The real ingress preserves Host and Uvicorn resolves HTTPS, but the platform
        # pattern names the private Kubernetes service, not this OAuth resource.
        monkeypatch.setenv(
            "INTERNAL_APP_ENDPOINT_PATTERN",
            "http://{app_fqdn}.chrismatteson-agents-development.svc.cluster.local",
        )
        resource = "https://icy-voice-1db2f.apps.demo.hosted.unionai.cloud/flyte-mcp/mcp"
        client = _client(_env(), base_url=resource.removesuffix("/flyte-mcp/mcp"))
        headers = {
            "X-Forwarded-Host": "attacker.invalid",
            "X-Forwarded-Proto": "https",
            "Forwarded": "host=attacker.invalid;proto=http",
        }
        for path in (ROOT, SUFFIXED):
            response = client.get(path, headers=headers)
            assert response.status_code == 200
            assert response.json()["resource"] == resource
            assert response.headers["cache-control"] == "no-store"
        response = client.post("/flyte-mcp/mcp", headers=headers)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == (
            'Bearer resource_metadata="https://icy-voice-1db2f.apps.demo.hosted.unionai.cloud'
            '/.well-known/oauth-protected-resource/flyte-mcp/mcp"'
        )

    @pytest.mark.parametrize("peer, scheme", [("127.0.0.1", "https"), ("192.0.2.1", "http")])
    @pytest.mark.asyncio
    async def test_forwarded_scheme_uses_uvicorn_trusted_proxy_boundary(self, monkeypatch, peer, scheme):
        from httpx import ASGITransport, AsyncClient
        from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

        monkeypatch.setenv("INTERNAL_APP_ENDPOINT_PATTERN", "http://{app_fqdn}.svc.cluster.local")
        app = ProxyHeadersMiddleware(_env()._starlette_app, trusted_hosts="127.0.0.1")
        resource = f"{scheme}://public.example/flyte-mcp/mcp"
        async with AsyncClient(
            transport=ASGITransport(app=app, client=(peer, 12345)), base_url="http://public.example"
        ) as client:
            headers = {
                "X-Forwarded-Proto": "https",
                "X-Forwarded-Host": "attacker.invalid",
                "Forwarded": "host=attacker.invalid;proto=https",
            }
            for path in (ROOT, SUFFIXED):
                response = await client.get(path, headers=headers)
                assert response.status_code == 200
                assert response.json()["resource"] == resource
                assert response.headers["cache-control"] == "no-store"
            response = await client.post("/flyte-mcp/mcp", headers=headers)
            assert response.status_code == 401
            assert response.headers["www-authenticate"] == (
                f'Bearer resource_metadata="{scheme}://public.example{SUFFIXED}"'
            )


class TestDefensiveValidation:
    """Inputs only a misconfiguration or a misbehaving server can produce."""

    def test_unparseable_documentation_url_is_rejected(self):
        with pytest.raises(ValueError, match="resource_documentation must be an absolute http"):
            ProtectedResourceMetadata(authorization_servers=[AUTH_SERVER], resource_documentation="http://[::1")

    def test_challenge_middleware_rejects_invalid_scope_tokens(self):
        from flyte.ai.mcp._protected_resource import BearerChallengeMiddleware

        with pytest.raises(ValueError, match="valid OAuth scope tokens"):
            BearerChallengeMiddleware(
                app=lambda *_: None,
                resource_url=lambda _: "https://mcp.example.com/flyte-mcp/mcp",
                excluded_paths=set(),
                scopes=["read write"],
            )

    def test_resource_is_not_taken_from_the_socket_address(self):
        from starlette.datastructures import Headers

        env = _env()
        # Starlette builds the URL from the socket address when it cannot use Host; that
        # address must never become the advertised resource.
        request = SimpleNamespace(
            headers=Headers({"host": "mcp.example.com"}),
            url=SimpleNamespace(netloc="10.0.0.7:8080", scheme="http"),
        )
        with pytest.raises(ValueError, match="Invalid Host header"):
            env._resolve_resource_url(request)

    @pytest.mark.parametrize(("transport", "suffix"), [("sse", "/sse"), ("stdio", "")])
    def test_endpoint_path_follows_the_transport(self, transport, suffix):
        from mcp.server.fastmcp import FastMCP

        from flyte.ai.mcp import MCPAppEnvironment

        env = MCPAppEnvironment(name="t-mcp", mcp=FastMCP(name="unit-test"), transport=transport)
        assert env._mcp_endpoint_path == f"{env.mcp_mount_path}{suffix}"
