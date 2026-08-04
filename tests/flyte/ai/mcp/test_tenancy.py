"""Tests for multi-tenant central-mode routing (``flyte.ai.mcp._tenancy``).

Everything here is offline: ``ClientSet.for_api_key`` / ``for_endpoint`` are monkeypatched to
return sentinel objects, so no control plane is ever contacted.
"""

import asyncio
import base64
from pathlib import Path

import pytest

from flyte._initialize import _get_init_config, _InitConfig
from flyte.ai.mcp._tenancy import (
    DEFAULT_ALLOWED_ENDPOINT_SUFFIXES,
    ENDPOINT_HEADER,
    CentralTenantMiddleware,
    ClientCache,
    default_allowed_endpoint_suffixes,
    endpoint_allowed,
    endpoint_hostname,
)

SUFFIXES = [".hosted.unionai.cloud"]


def _api_key(endpoint: str, org: str = "acme") -> str:
    return base64.b64encode(f"{endpoint}:client-id:client-secret:{org}".encode()).decode()


class FakeClient:
    """Stand-in for ``ClientSet``; records whether it was closed."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_clientset(monkeypatch):
    """Monkeypatch both ``ClientSet`` constructors; returns the call log."""
    from flyte.remote._client.controlplane import ClientSet

    calls = {"api_key": [], "endpoint": []}

    async def fake_for_api_key(cls, api_key, **kwargs):
        calls["api_key"].append(api_key)
        return FakeClient(f"api-key:{len(calls['api_key'])}")

    async def fake_for_endpoint(cls, endpoint, **kwargs):
        calls["endpoint"].append((endpoint, kwargs))
        return FakeClient(endpoint)

    monkeypatch.setattr(ClientSet, "for_api_key", classmethod(fake_for_api_key))
    monkeypatch.setattr(ClientSet, "for_endpoint", classmethod(fake_for_endpoint))
    return calls


# ------------------------------
# Endpoint allowlist
# ------------------------------


class TestEndpointHostname:
    @pytest.mark.parametrize(
        "endpoint,expected",
        [
            ("foo.hosted.unionai.cloud", "foo.hosted.unionai.cloud"),
            ("dns:///foo.hosted.unionai.cloud", "foo.hosted.unionai.cloud"),
            ("https://foo.hosted.unionai.cloud", "foo.hosted.unionai.cloud"),
            ("http://foo.hosted.unionai.cloud:8080", "foo.hosted.unionai.cloud"),
            ("FOO.Hosted.UnionAI.Cloud.", "foo.hosted.unionai.cloud"),
            ("https://foo.hosted.unionai.cloud/some/path", "foo.hosted.unionai.cloud"),
            ("foo.hosted.unionai.cloud:443", "foo.hosted.unionai.cloud"),
            ("[::1]:8080", "::1"),
            ("", ""),
            ("   ", ""),
            ("ftp://foo.hosted.unionai.cloud", ""),
        ],
    )
    def test_normalization(self, endpoint, expected):
        assert endpoint_hostname(endpoint) == expected


class TestEndpointAllowed:
    @pytest.mark.parametrize(
        "endpoint",
        [
            "foo.hosted.unionai.cloud",
            "dns:///foo.hosted.unionai.cloud",
            "https://foo.hosted.unionai.cloud",
            "dns:///foo.bar.hosted.unionai.cloud",
            "foo.hosted.unionai.cloud:443",
        ],
    )
    def test_accepts_allowlisted(self, endpoint):
        assert endpoint_allowed(endpoint, SUFFIXES) is True

    @pytest.mark.parametrize(
        "endpoint",
        [
            "evil.com",
            "dns:///evil.com",
            "foo.hosted.unionai.cloud.evil.com",
            "https://foo.hosted.unionai.cloud.evil.com",
            "localhost",
            "localhost:8090",
            "dns:///localhost:30080",
            "app.localhost",
            "127.0.0.1",
            "dns:///127.0.0.1:8090",
            "169.254.169.254",
            "[::1]:8080",
            "",
            # The apex itself is not a tenant of the ``.suffix`` form.
            "hosted.unionai.cloud",
        ],
    )
    def test_rejects_everything_else(self, endpoint):
        assert endpoint_allowed(endpoint, SUFFIXES) is False

    def test_exact_entry_permits_localhost(self):
        assert endpoint_allowed("dns:///localhost:8090", ["localhost"]) is True
        assert endpoint_allowed("127.0.0.1:8090", ["127.0.0.1"]) is True

    def test_bare_suffix_matches_subdomains_only(self):
        assert endpoint_allowed("foo.example.com", ["example.com"]) is True
        assert endpoint_allowed("example.com", ["example.com"]) is True
        assert endpoint_allowed("notexample.com", ["example.com"]) is False

    def test_empty_allowlist_rejects(self):
        assert endpoint_allowed("foo.hosted.unionai.cloud", []) is False

    def test_default_suffixes_from_env(self, monkeypatch):
        monkeypatch.delenv("FLYTE_MCP_ALLOWED_ENDPOINT_SUFFIXES", raising=False)
        assert default_allowed_endpoint_suffixes() == list(DEFAULT_ALLOWED_ENDPOINT_SUFFIXES)

        monkeypatch.setenv("FLYTE_MCP_ALLOWED_ENDPOINT_SUFFIXES", " .a.example.com , .b.example.com ,")
        assert default_allowed_endpoint_suffixes() == [".a.example.com", ".b.example.com"]
        # ``suffixes=None`` picks up the env-configured list.
        assert endpoint_allowed("t.a.example.com") is True
        assert endpoint_allowed("t.hosted.unionai.cloud") is False


# ------------------------------
# Client cache
# ------------------------------


class TestClientCache:
    @pytest.mark.asyncio
    async def test_hit_returns_same_config_object(self, fake_clientset):
        cache = ClientCache()
        key = _api_key("foo.hosted.unionai.cloud")

        first = await cache.get_for_api_key(key, endpoint="dns:///foo.hosted.unionai.cloud", org="acme")
        second = await cache.get_for_api_key(key, endpoint="dns:///foo.hosted.unionai.cloud", org="acme")

        assert first is second
        assert len(fake_clientset["api_key"]) == 1
        assert len(cache) == 1

    @pytest.mark.asyncio
    async def test_config_carries_org_and_client(self, fake_clientset):
        cache = ClientCache(root_dir=Path("/tmp"))
        cfg = await cache.get_for_api_key(_api_key("foo.hosted.unionai.cloud"), endpoint="x", org="acme")

        assert isinstance(cfg, _InitConfig)
        assert cfg.org == "acme"
        assert isinstance(cfg.client, FakeClient)
        # A central server has no single default project/domain.
        assert cfg.project is None
        assert cfg.domain is None

    @pytest.mark.asyncio
    async def test_distinct_keys_get_distinct_configs(self, fake_clientset):
        cache = ClientCache()
        a = await cache.get_for_api_key(_api_key("a.hosted.unionai.cloud"), endpoint="a", org="a")
        b = await cache.get_for_api_key(_api_key("b.hosted.unionai.cloud"), endpoint="b", org="b")

        assert a is not b
        assert len(fake_clientset["api_key"]) == 2

    @pytest.mark.asyncio
    async def test_endpoint_path_uses_passthrough_auth(self, fake_clientset):
        cache = ClientCache()
        cfg = await cache.get_for_endpoint("foo.hosted.unionai.cloud")

        endpoint, kwargs = fake_clientset["endpoint"][0]
        assert endpoint == "dns:///foo.hosted.unionai.cloud"
        assert kwargs["auth_type"] == "Passthrough"
        # org is derived from the endpoint so console URLs resolve to the right tenant.
        assert cfg.org == "foo"

        # Keyed by endpoint, so a second (differently-tokened) caller reuses the client.
        assert await cache.get_for_endpoint("https://foo.hosted.unionai.cloud") is cfg
        assert len(fake_clientset["endpoint"]) == 1

    @pytest.mark.asyncio
    async def test_lru_eviction_closes_client(self, fake_clientset):
        cache = ClientCache(max_entries=2)
        first = await cache.get_for_api_key(_api_key("a.hosted.unionai.cloud"), endpoint="a", org="a")
        await cache.get_for_api_key(_api_key("b.hosted.unionai.cloud"), endpoint="b", org="b")
        await cache.get_for_api_key(_api_key("c.hosted.unionai.cloud"), endpoint="c", org="c")

        assert len(cache) == 2
        assert first.client.closed is True

    @pytest.mark.asyncio
    async def test_ttl_expiry_forces_rebuild(self, fake_clientset):
        cache = ClientCache(ttl_s=-1.0)  # everything is immediately stale
        key = _api_key("a.hosted.unionai.cloud")
        first = await cache.get_for_api_key(key, endpoint="a", org="a")
        second = await cache.get_for_api_key(key, endpoint="a", org="a")

        assert first is not second
        assert len(fake_clientset["api_key"]) == 2

    @pytest.mark.asyncio
    async def test_concurrent_misses_build_once(self, monkeypatch):
        from flyte.remote._client.controlplane import ClientSet

        calls = []

        async def slow_for_api_key(cls, api_key, **kwargs):
            calls.append(api_key)
            await asyncio.sleep(0.01)
            return FakeClient("slow")

        monkeypatch.setattr(ClientSet, "for_api_key", classmethod(slow_for_api_key))

        cache = ClientCache()
        key = _api_key("a.hosted.unionai.cloud")
        results = await asyncio.gather(*[cache.get_for_api_key(key, endpoint="a", org="a") for _ in range(5)])

        assert len(calls) == 1
        assert all(r is results[0] for r in results)


# ------------------------------
# Middleware
# ------------------------------


def _build_app(**middleware_kwargs):
    """A Starlette app whose probe route reports the init-config the request is scoped to."""
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def probe(_request):
        from flyte.remote._auth_metadata import get_auth_metadata

        cfg = _get_init_config()
        return JSONResponse(
            {
                "org": None if cfg is None else cfg.org,
                "cfg_id": None if cfg is None else id(cfg),
                "client_endpoint": None if cfg is None else getattr(cfg.client, "endpoint", None),
                "auth_metadata": [list(kv) for kv in (get_auth_metadata() or ())],
            }
        )

    async def health(_request):
        return JSONResponse({"status": "healthy"})

    middleware_kwargs.setdefault("allowed_endpoint_suffixes", SUFFIXES)
    return Starlette(
        routes=[Route("/probe", probe), Route("/health", health), Route("/", health)],
        middleware=[Middleware(CentralTenantMiddleware, **middleware_kwargs)],
    )


@pytest.fixture
def client_factory(fake_clientset):
    from starlette.testclient import TestClient

    async def _accept():
        return None

    def make(**middleware_kwargs):
        # Credential verification hits the control plane; tests inject a stub unless they are
        # specifically exercising a rejection.
        middleware_kwargs.setdefault("verifier", _accept)
        middleware_kwargs.setdefault("excluded_paths", {"/health", "/"})
        return TestClient(_build_app(**middleware_kwargs), raise_server_exceptions=False)

    return make


class TestCentralTenantMiddleware:
    def test_missing_authorization_is_401(self, client_factory):
        resp = client_factory().get("/probe")
        assert resp.status_code == 401
        assert resp.headers["WWW-Authenticate"] == "Bearer"
        # The 401 tells the caller how to get a credential.
        assert "api key" in resp.json()["detail"].lower()

    def test_non_bearer_scheme_is_401(self, client_factory):
        resp = client_factory().get("/probe", headers={"Authorization": "Basic abc"})
        assert resp.status_code == 401

    def test_excluded_paths_need_no_credential(self, client_factory):
        client = client_factory()
        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 200

    def test_excluded_paths_are_configurable(self, client_factory):
        # With only /health excluded, "/" now requires a credential.
        client = client_factory(excluded_paths={"/health"})
        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 401

    def test_api_key_enters_tenant_context(self, client_factory, fake_clientset):
        key = _api_key("foo.hosted.unionai.cloud", org="acme")
        resp = client_factory().get("/probe", headers={"Authorization": f"Bearer {key}"})

        assert resp.status_code == 200
        assert resp.json()["org"] == "acme"
        assert fake_clientset["api_key"] == [key]

    def test_api_key_with_no_org_falls_back_to_endpoint_org(self, client_factory):
        # Keys minted without an org (4th field "None") must still work for org-scoped
        # calls: the org is derived from the endpoint hostname, same as the bearer path.
        key = base64.b64encode(b"foo.hosted.unionai.cloud:cid:secret:None").decode()
        resp = client_factory().get("/probe", headers={"Authorization": f"Bearer {key}"})

        assert resp.status_code == 200
        assert resp.json()["org"] == "foo"

    def test_context_does_not_leak_between_requests(self, client_factory):
        client = client_factory()
        key = _api_key("foo.hosted.unionai.cloud", org="acme")
        assert client.get("/probe", headers={"Authorization": f"Bearer {key}"}).json()["org"] == "acme"
        # No credential -> 401; and an excluded path sees no tenant config at all.
        assert client.get("/health").status_code == 200

    def test_two_tenants_get_two_configs(self, client_factory, fake_clientset):
        client = client_factory()
        a = client.get("/probe", headers={"Authorization": f"Bearer {_api_key('a.hosted.unionai.cloud', 'a')}"})
        b = client.get("/probe", headers={"Authorization": f"Bearer {_api_key('b.hosted.unionai.cloud', 'b')}"})

        assert a.json()["org"] == "a"
        assert b.json()["org"] == "b"
        assert a.json()["cfg_id"] != b.json()["cfg_id"]

    def test_cache_hit_returns_same_config(self, client_factory, fake_clientset):
        client = client_factory()
        key = _api_key("foo.hosted.unionai.cloud")
        first = client.get("/probe", headers={"Authorization": f"Bearer {key}"}).json()
        second = client.get("/probe", headers={"Authorization": f"Bearer {key}"}).json()

        assert first["cfg_id"] == second["cfg_id"]
        assert len(fake_clientset["api_key"]) == 1

    def test_disallowed_endpoint_is_403(self, client_factory, fake_clientset):
        key = _api_key("foo.hosted.unionai.cloud.evil.com")
        resp = client_factory().get("/probe", headers={"Authorization": f"Bearer {key}"})

        assert resp.status_code == 403
        detail = resp.json()["detail"]
        assert "foo.hosted.unionai.cloud.evil.com" in detail
        # Only the hostname is echoed; no credential material.
        assert key not in detail
        assert "client-secret" not in detail
        assert fake_clientset["api_key"] == []

    def test_bearer_without_endpoint_header_is_401(self, client_factory, fake_clientset):
        resp = client_factory().get("/probe", headers={"Authorization": "Bearer not-an-api-key"})

        assert resp.status_code == 401
        assert ENDPOINT_HEADER in resp.json()["detail"].lower()
        assert fake_clientset["endpoint"] == []

    def test_bearer_with_endpoint_header_enters_context(self, client_factory, fake_clientset):
        resp = client_factory().get(
            "/probe",
            headers={"Authorization": "Bearer raw-token", ENDPOINT_HEADER: "foo.hosted.unionai.cloud"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["client_endpoint"] == "dns:///foo.hosted.unionai.cloud"
        # The caller's token is forwarded verbatim to the control plane.
        assert ["authorization", "Bearer raw-token"] in body["auth_metadata"]

    def test_bearer_with_disallowed_endpoint_header_is_403(self, client_factory, fake_clientset):
        resp = client_factory().get(
            "/probe",
            headers={"Authorization": "Bearer raw-token", ENDPOINT_HEADER: "evil.com"},
        )

        assert resp.status_code == 403
        assert fake_clientset["endpoint"] == []

    def test_client_construction_failure_is_502(self, client_factory, monkeypatch):
        from flyte.remote._client.controlplane import ClientSet

        async def boom(cls, api_key, **kwargs):
            raise RuntimeError(f"handshake failed for {api_key}")

        monkeypatch.setattr(ClientSet, "for_api_key", classmethod(boom))

        key = _api_key("foo.hosted.unionai.cloud")
        resp = client_factory().get("/probe", headers={"Authorization": f"Bearer {key}"})

        assert resp.status_code == 502
        # The underlying exception can embed the credential, so it must not be echoed.
        assert key not in resp.json()["detail"]

    def test_custom_decoder_is_used(self, client_factory, fake_clientset):
        def decoder(cred):
            assert cred == "opaque"
            return ("dns:///tenant.hosted.unionai.cloud", "cid", "secret", "tenant-org")

        resp = client_factory(decoder=decoder).get("/probe", headers={"Authorization": "Bearer opaque"})

        assert resp.status_code == 200
        assert resp.json()["org"] == "tenant-org"

    def test_shared_cache_across_middleware_instances(self, fake_clientset):
        from starlette.testclient import TestClient

        cache = ClientCache()
        key = _api_key("foo.hosted.unionai.cloud")
        headers = {"Authorization": f"Bearer {key}"}

        async def _accept():
            return None

        a = TestClient(_build_app(cache=cache, verifier=_accept), raise_server_exceptions=False).get(
            "/probe", headers=headers
        )
        b = TestClient(_build_app(cache=cache, verifier=_accept), raise_server_exceptions=False).get(
            "/probe", headers=headers
        )

        assert a.json()["cfg_id"] == b.json()["cfg_id"]
        assert len(fake_clientset["api_key"]) == 1

    def test_credentials_are_not_logged(self, client_factory):
        # Attached directly to the module logger rather than via caplog: flyte's logging setup
        # can stop records from propagating to the root handler caplog installs.
        import logging

        records: list[str] = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        logger = logging.getLogger("flyte.ai.mcp._tenancy")
        handler = Capture(level=logging.DEBUG)
        previous_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            key = _api_key("foo.hosted.unionai.cloud", org="acme")
            client_factory().get("/probe", headers={"Authorization": f"Bearer {key}"})
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

        text = "\n".join(records)
        assert key not in text
        assert "client-secret" not in text
        # ...but the org, host and path are logged for operability.
        assert "acme" in text
        assert "foo.hosted.unionai.cloud" in text
        assert "/probe" in text


class TestCredentialVerification:
    """A credential must be proven against its control plane before any tool runs.

    Decoding an API key is pure string work and building a ClientSet does no I/O, so without
    an explicit check a forged credential reaches the tools — on a public endpoint that means
    anonymous access to every non-control-plane tool (the search corpus).
    """

    @pytest.fixture
    def rejecting_client(self, fake_clientset):
        from starlette.testclient import TestClient

        async def _reject():
            raise PermissionError("invalid credential")

        return TestClient(_build_app(verifier=_reject), raise_server_exceptions=False)

    def test_forged_api_key_is_rejected(self, rejecting_client):
        forged = _api_key("foo.hosted.unionai.cloud", org="anyorg")
        resp = rejecting_client.get("/probe", headers={"Authorization": f"Bearer {forged}"})

        assert resp.status_code == 401
        assert resp.headers["WWW-Authenticate"] == "Bearer"

    def test_forged_bearer_token_is_rejected(self, rejecting_client):
        resp = rejecting_client.get(
            "/probe",
            headers={
                "Authorization": "Bearer not-a-real-token",
                ENDPOINT_HEADER: "foo.hosted.unionai.cloud",
            },
        )

        assert resp.status_code == 401

    def test_verification_is_cached_per_credential(self, fake_clientset):
        from starlette.testclient import TestClient

        calls = {"n": 0}

        async def _count():
            calls["n"] += 1

        client = TestClient(_build_app(verifier=_count), raise_server_exceptions=False)
        headers = {"Authorization": f"Bearer {_api_key('foo.hosted.unionai.cloud')}"}
        for _ in range(3):
            assert client.get("/probe", headers=headers).status_code == 200

        assert calls["n"] == 1, "verification should ride the TTL cache after the first request"


class TestEndpointSpoofing:
    """endpoint_allowed must reject anything whose real connect target differs from the
    string it matched — a URL fragment is dropped by urlparse at connect time."""

    @pytest.mark.parametrize(
        "endpoint",
        [
            "evil.com#foo.hosted.unionai.cloud",
            "evil.com:443#foo.hosted.unionai.cloud",
            "169.254.169.254#x.hosted.unionai.cloud",
            "localhost#x.hosted.unionai.cloud",
            "foo\nINJECTED.hosted.unionai.cloud",
            "foo bar.hosted.unionai.cloud",
        ],
    )
    def test_spoofed_endpoints_are_rejected(self, endpoint):
        assert endpoint_allowed(endpoint, [".hosted.unionai.cloud"]) is False

    @pytest.mark.parametrize(
        "endpoint",
        [
            "ok.hosted.unionai.cloud",
            "dns:///ok.hosted.unionai.cloud",
            "https://ok.hosted.unionai.cloud:443",
            "OK.Hosted.UnionAI.Cloud.",
        ],
    )
    def test_legitimate_spellings_still_pass(self, endpoint):
        assert endpoint_allowed(endpoint, [".hosted.unionai.cloud"]) is True
