"""Tests for multi-tenant central-mode routing (``flyte.ai.mcp._tenancy``).

Everything here is offline: ``ClientSet.for_api_key`` / ``for_endpoint`` and
``RemoteClientConfigStore`` are monkeypatched to return sentinel objects, so no control plane
is ever contacted.
"""

import asyncio
import base64
from pathlib import Path

import pytest

from flyte._initialize import _get_init_config, _InitConfig
from flyte.ai.mcp._tenancy import (
    DEFAULT_ALLOWED_ENDPOINT_PATTERNS,
    ENDPOINT_HEADER,
    RESERVED_ORG_LABELS,
    CentralTenantMiddleware,
    ClientCache,
    RateLimiter,
    TokenEndpointNotAllowed,
    configured_endpoint_suffixes,
    endpoint_allowed,
    endpoint_hostname,
    extra_allowed_token_endpoint_suffixes,
)

SUFFIXES = [".hosted.unionai.cloud"]

#: A token endpoint under the tenant domain — what a well-behaved control plane advertises.
GOOD_TOKEN_ENDPOINT = "https://foo.hosted.unionai.cloud/oauth2/token"


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
def fake_config_store(monkeypatch):
    """Stub ``RemoteClientConfigStore`` so the token-endpoint check never hits the network.

    Mutate ``["token_endpoint"]`` to make the control plane advertise somewhere else;
    ``["addresses"]`` records which endpoints were asked for their OAuth2 metadata.
    """
    from flyte.remote._client.auth import _client_config

    state: dict = {"token_endpoint": GOOD_TOKEN_ENDPOINT, "addresses": []}

    class FakeStore:
        def __init__(self, endpoint, http_client=None):
            state["addresses"].append(endpoint)

        async def get_client_config(self):
            return _client_config.ClientConfig(
                token_endpoint=state["token_endpoint"],
                authorization_endpoint="https://foo.hosted.unionai.cloud/oauth2/authorize",
                redirect_uri="http://localhost:53593/callback",
                client_id="flytepropeller",
            )

    monkeypatch.setattr(_client_config, "RemoteClientConfigStore", FakeStore)
    return state


@pytest.fixture
def fake_clientset(monkeypatch, fake_config_store):
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

    def test_env_replaces_the_default_patterns(self, monkeypatch):
        monkeypatch.delenv("FLYTE_MCP_ALLOWED_ENDPOINT_SUFFIXES", raising=False)
        # No env var -> no operator override -> the default control-plane patterns apply.
        assert configured_endpoint_suffixes() is None
        assert endpoint_allowed("t.hosted.unionai.cloud") is True

        monkeypatch.setenv("FLYTE_MCP_ALLOWED_ENDPOINT_SUFFIXES", " .a.example.com , .b.example.com ,")
        assert configured_endpoint_suffixes() == [".a.example.com", ".b.example.com"]
        # ``suffixes=None`` picks up the env-configured list, which *replaces* the defaults:
        # a Union control plane is no longer reachable unless the operator named it.
        assert endpoint_allowed("t.a.example.com") is True
        assert endpoint_allowed("t.hosted.unionai.cloud") is False

    def test_env_escape_hatch_admits_an_unlisted_zone(self, monkeypatch):
        # A zone outside the default patterns is off until an operator names it. That opt-in
        # has to actually work, since it is the only route for a self-hosted deployment.
        assert endpoint_allowed("acme.sh-us-east-2.unionai.cloud") is False
        monkeypatch.setenv("FLYTE_MCP_ALLOWED_ENDPOINT_SUFFIXES", ".sh-us-east-2.unionai.cloud")
        assert endpoint_allowed("acme.sh-us-east-2.unionai.cloud") is True


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
    async def test_concurrent_misses_build_once(self, monkeypatch, fake_config_store):
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


class TestRateLimiter:
    """The token bucket itself, independent of any HTTP plumbing."""

    def test_burst_then_throttled(self):
        limiter = RateLimiter(rpm=60, burst=3)
        assert [limiter.check("k") for _ in range(3)] == [None, None, None]
        retry_after = limiter.check("k")
        assert isinstance(retry_after, int)
        assert retry_after >= 1

    def test_keys_are_independent(self):
        limiter = RateLimiter(rpm=60, burst=1)
        assert limiter.check("a") is None
        assert limiter.check("a") is not None
        # A second tenant's credential is untouched by the first one's exhaustion.
        assert limiter.check("b") is None

    def test_zero_rpm_disables(self):
        limiter = RateLimiter(rpm=0, burst=1)
        assert limiter.enabled is False
        assert all(limiter.check("k") is None for _ in range(50))
        assert len(limiter) == 0

    def test_env_configures_defaults(self, monkeypatch):
        monkeypatch.setenv("FLYTE_MCP_RATE_LIMIT_RPM", "30")
        monkeypatch.setenv("FLYTE_MCP_RATE_LIMIT_BURST", "2")
        limiter = RateLimiter()
        assert (limiter.rpm, limiter.burst) == (30, 2)
        assert [limiter.check("k") for _ in range(3)] == [None, None, 2]

    def test_env_zero_disables(self, monkeypatch):
        monkeypatch.setenv("FLYTE_MCP_RATE_LIMIT_RPM", "0")
        assert RateLimiter().enabled is False

    def test_defaults_when_env_absent_or_junk(self, monkeypatch):
        monkeypatch.delenv("FLYTE_MCP_RATE_LIMIT_RPM", raising=False)
        monkeypatch.delenv("FLYTE_MCP_RATE_LIMIT_BURST", raising=False)
        assert (RateLimiter().rpm, RateLimiter().burst) == (120, 30)

        monkeypatch.setenv("FLYTE_MCP_RATE_LIMIT_RPM", "not-a-number")
        monkeypatch.setenv("FLYTE_MCP_RATE_LIMIT_BURST", "-5")
        assert (RateLimiter().rpm, RateLimiter().burst) == (120, 30)

    def test_constructor_args_beat_env(self, monkeypatch):
        monkeypatch.setenv("FLYTE_MCP_RATE_LIMIT_RPM", "999")
        assert RateLimiter(rpm=7, burst=1).rpm == 7

    def test_tokens_refill_over_time(self, monkeypatch):
        import flyte.ai.mcp._tenancy as tenancy

        clock = {"t": 1000.0}
        monkeypatch.setattr(tenancy.time, "monotonic", lambda: clock["t"])

        limiter = RateLimiter(rpm=60, burst=1)  # one token per second
        assert limiter.check("k") is None
        assert limiter.check("k") == 1
        clock["t"] += 1.0
        assert limiter.check("k") is None

    def test_idle_buckets_are_evicted(self, monkeypatch):
        import flyte.ai.mcp._tenancy as tenancy

        clock = {"t": 1000.0}
        monkeypatch.setattr(tenancy.time, "monotonic", lambda: clock["t"])

        limiter = RateLimiter(rpm=60, burst=2)
        limiter.check("k")
        assert len(limiter) == 1
        # Past a full refill the bucket carries no state, so it is dropped rather than kept.
        clock["t"] += 10.0
        limiter.check("other")
        assert len(limiter) == 1

    def test_bucket_count_is_capped(self):
        limiter = RateLimiter(rpm=60, burst=1, max_buckets=4)
        for i in range(50):
            limiter.check(f"k{i}")
        assert len(limiter) <= 4


class TestMiddlewareRateLimiting:
    """A shared public endpoint must not let one tenant's runaway agent starve the rest."""

    def _headers(self, endpoint="foo.hosted.unionai.cloud", org="acme"):
        return {"Authorization": f"Bearer {_api_key(endpoint, org)}"}

    def test_429_once_the_burst_is_spent(self, client_factory):
        client = client_factory(rate_limiter=RateLimiter(rpm=60, burst=2))
        headers = self._headers()

        assert client.get("/probe", headers=headers).status_code == 200
        assert client.get("/probe", headers=headers).status_code == 200

        resp = client.get("/probe", headers=headers)
        assert resp.status_code == 429
        assert int(resp.headers["Retry-After"]) >= 1
        detail = resp.json()["detail"]
        assert "60 requests/minute" in detail
        assert "burst 2" in detail

    def test_other_credentials_are_unaffected(self, client_factory):
        client = client_factory(rate_limiter=RateLimiter(rpm=60, burst=1))
        noisy = self._headers("noisy.hosted.unionai.cloud", "noisy")

        assert client.get("/probe", headers=noisy).status_code == 200
        assert client.get("/probe", headers=noisy).status_code == 429
        # A different tenant's key hashes to a different bucket.
        quiet = self._headers("quiet.hosted.unionai.cloud", "quiet")
        assert client.get("/probe", headers=quiet).status_code == 200

    def test_disabled_by_zero(self, client_factory):
        client = client_factory(rate_limiter=RateLimiter(rpm=0))
        headers = self._headers()
        assert all(client.get("/probe", headers=headers).status_code == 200 for _ in range(20))

    def test_excluded_paths_are_exempt(self, client_factory):
        client = client_factory(rate_limiter=RateLimiter(rpm=60, burst=1))
        headers = self._headers()

        assert client.get("/probe", headers=headers).status_code == 200
        assert client.get("/probe", headers=headers).status_code == 429
        # Liveness probes are unauthenticated and must never be throttled away.
        for _ in range(10):
            assert client.get("/health", headers=headers).status_code == 200
            assert client.get("/health").status_code == 200

    def test_throttle_precedes_client_construction(self, client_factory, fake_clientset):
        """A throttled caller must not cost a ClientSet build or a verification round-trip."""
        client = client_factory(rate_limiter=RateLimiter(rpm=60, burst=1))
        headers = self._headers()

        assert client.get("/probe", headers=headers).status_code == 200
        for _ in range(5):
            assert client.get("/probe", headers=headers).status_code == 429
        assert len(fake_clientset["api_key"]) == 1

    def test_bad_credentials_are_throttled_too(self, client_factory, fake_clientset):
        # An endpoint that fails the allowlist still spends a token, so a forged-credential
        # flood cannot run unbounded just because it never reaches a client.
        client = client_factory(rate_limiter=RateLimiter(rpm=60, burst=1))
        headers = {"Authorization": f"Bearer {_api_key('evil.com')}"}

        assert client.get("/probe", headers=headers).status_code == 403
        assert client.get("/probe", headers=headers).status_code == 429

    def test_default_limiter_allows_normal_traffic(self, client_factory):
        # The env-configured default (120rpm/30burst) must not trip on ordinary use.
        client = client_factory()
        headers = self._headers()
        assert all(client.get("/probe", headers=headers).status_code == 200 for _ in range(10))


class TestTokenEndpointAllowlist:
    """flyte's client-credentials flow POSTs the tenant's secret to whatever ``token_endpoint``
    the target control plane advertises. An allowlisted-but-malicious control plane could
    therefore aim that POST anywhere, so the advertised URL is checked before the client is used.
    """

    @pytest.mark.asyncio
    async def test_off_allowlist_token_endpoint_raises(self, fake_clientset, fake_config_store):
        fake_config_store["token_endpoint"] = "https://evil.com/oauth2/token"
        cache = ClientCache(allowed_endpoint_suffixes=SUFFIXES)

        with pytest.raises(TokenEndpointNotAllowed) as excinfo:
            await cache.get_for_api_key(
                _api_key("foo.hosted.unionai.cloud"), endpoint="dns:///foo.hosted.unionai.cloud", org="acme"
            )

        assert "evil.com" in str(excinfo.value)
        # Nothing is cached for a rejected tenant.
        assert len(cache) == 0

    @pytest.mark.asyncio
    async def test_on_allowlist_token_endpoint_is_accepted(self, fake_clientset, fake_config_store):
        cache = ClientCache(allowed_endpoint_suffixes=SUFFIXES)
        cfg = await cache.get_for_api_key(
            _api_key("foo.hosted.unionai.cloud"), endpoint="dns:///foo.hosted.unionai.cloud", org="acme"
        )
        assert isinstance(cfg, _InitConfig)
        # The metadata is fetched from the control plane the client actually dials.
        assert fake_config_store["addresses"] == ["dns:///foo.hosted.unionai.cloud"]

    @pytest.mark.asyncio
    async def test_escape_hatch_env_accepts_external_idp(self, fake_clientset, fake_config_store, monkeypatch):
        fake_config_store["token_endpoint"] = "https://acme.okta.com/oauth2/v1/token"
        cache = ClientCache(allowed_endpoint_suffixes=SUFFIXES)

        with pytest.raises(TokenEndpointNotAllowed):
            await cache.get_for_api_key(_api_key("foo.hosted.unionai.cloud"), endpoint="foo", org="acme")

        monkeypatch.setenv("FLYTE_MCP_ALLOWED_TOKEN_ENDPOINT_SUFFIXES", ".okta.com")
        assert extra_allowed_token_endpoint_suffixes() == [".okta.com"]
        cfg = await ClientCache(allowed_endpoint_suffixes=SUFFIXES).get_for_api_key(
            _api_key("foo.hosted.unionai.cloud"), endpoint="foo", org="acme"
        )
        assert cfg.org == "acme"

    @pytest.mark.asyncio
    async def test_empty_token_endpoint_is_rejected(self, fake_clientset, fake_config_store):
        fake_config_store["token_endpoint"] = ""
        with pytest.raises(TokenEndpointNotAllowed):
            await ClientCache(allowed_endpoint_suffixes=SUFFIXES).get_for_api_key(
                _api_key("foo.hosted.unionai.cloud"), endpoint="foo", org="acme"
            )

    @pytest.mark.asyncio
    async def test_validation_can_be_switched_off(self, fake_clientset, fake_config_store):
        fake_config_store["token_endpoint"] = "https://evil.com/oauth2/token"
        cache = ClientCache(allowed_endpoint_suffixes=SUFFIXES, validate_token_endpoint=False)
        assert await cache.get_for_api_key(_api_key("foo.hosted.unionai.cloud"), endpoint="foo", org="acme")
        assert fake_config_store["addresses"] == []

    @pytest.mark.asyncio
    async def test_passthrough_path_does_no_token_exchange(self, fake_clientset, fake_config_store):
        # PassthroughAuthenticator never resolves a client config or calls get_token, so the
        # raw-bearer path has no credential POST to redirect and needs no check.
        cache = ClientCache(allowed_endpoint_suffixes=SUFFIXES)
        await cache.get_for_endpoint("foo.hosted.unionai.cloud")
        assert fake_config_store["addresses"] == []

    def test_middleware_renders_rejection_as_403(self, client_factory, fake_config_store):
        fake_config_store["token_endpoint"] = "https://evil.com/oauth2/token"
        key = _api_key("foo.hosted.unionai.cloud")

        resp = client_factory().get("/probe", headers={"Authorization": f"Bearer {key}"})

        assert resp.status_code == 403
        detail = resp.json()["detail"]
        assert "evil.com" in detail
        # Only hostnames the control plane published are echoed; no credential material.
        assert key not in detail
        assert "client-secret" not in detail

    def test_middleware_serves_an_on_allowlist_token_endpoint(self, client_factory, fake_config_store):
        resp = client_factory().get(
            "/probe", headers={"Authorization": f"Bearer {_api_key('foo.hosted.unionai.cloud')}"}
        )
        assert resp.status_code == 200

    def test_middleware_escape_hatch_env(self, client_factory, fake_config_store, monkeypatch):
        fake_config_store["token_endpoint"] = "https://acme.okta.com/oauth2/v1/token"
        monkeypatch.setenv("FLYTE_MCP_ALLOWED_TOKEN_ENDPOINT_SUFFIXES", ".okta.com")

        resp = client_factory().get(
            "/probe", headers={"Authorization": f"Bearer {_api_key('foo.hosted.unionai.cloud')}"}
        )
        assert resp.status_code == 200

    def test_middleware_passes_its_allowlist_to_the_cache(self, fake_config_store):
        # The middleware's endpoint allowlist is what the token endpoint is judged against, so a
        # deployment that narrows one narrows the other.
        mw = CentralTenantMiddleware(app=None, allowed_endpoint_suffixes=[".example.com"])
        assert mw.cache._allowed_endpoint_suffixes == [".example.com"]
        assert mw.cache._token_endpoint_allowed("https://idp.example.com/oauth2/token") is True
        assert mw.cache._token_endpoint_allowed("https://foo.hosted.unionai.cloud/oauth2/token") is False

    @pytest.mark.asyncio
    async def test_default_patterns_accept_a_control_plane_token_endpoint(self, fake_clientset, fake_config_store):
        # With no configured suffixes the token endpoint is judged against the default
        # control-plane patterns, so a tenant hosting its own /oauth2/token still works...
        cache = ClientCache()
        assert await cache.get_for_api_key(
            _api_key("foo.hosted.unionai.cloud"), endpoint="dns:///foo.hosted.unionai.cloud", org="acme"
        )

        # ...while a redirect to anything off the default set is refused.
        fake_config_store["token_endpoint"] = "https://evil.sh-us-east-2.unionai.cloud/oauth2/token"
        with pytest.raises(TokenEndpointNotAllowed):
            await ClientCache().get_for_api_key(
                _api_key("bar.hosted.unionai.cloud"), endpoint="dns:///bar.hosted.unionai.cloud", org="acme"
            )

    @pytest.mark.asyncio
    async def test_extra_suffixes_apply_in_default_pattern_mode(self, fake_clientset, fake_config_store, monkeypatch):
        # The extras are checked separately from the defaults rather than concatenated onto
        # them, so an external IdP can be admitted without an explicit endpoint allowlist.
        fake_config_store["token_endpoint"] = "https://acme.okta.com/oauth2/v1/token"
        with pytest.raises(TokenEndpointNotAllowed):
            await ClientCache().get_for_api_key(
                _api_key("foo.hosted.unionai.cloud"), endpoint="dns:///foo.hosted.unionai.cloud", org="acme"
            )

        monkeypatch.setenv("FLYTE_MCP_ALLOWED_TOKEN_ENDPOINT_SUFFIXES", ".okta.com")
        assert await ClientCache().get_for_api_key(
            _api_key("foo.hosted.unionai.cloud"), endpoint="dns:///foo.hosted.unionai.cloud", org="acme"
        )

    def test_extra_suffixes_default_to_empty(self, monkeypatch):
        monkeypatch.delenv("FLYTE_MCP_ALLOWED_TOKEN_ENDPOINT_SUFFIXES", raising=False)
        assert extra_allowed_token_endpoint_suffixes() == []
        monkeypatch.setenv("FLYTE_MCP_ALLOWED_TOKEN_ENDPOINT_SUFFIXES", " .a.com , .b.com ,")
        assert extra_allowed_token_endpoint_suffixes() == [".a.com", ".b.com"]


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


class TestDefaultAllowlistCoversRealTenants:
    """The default allowlist is an explicit set of control-plane shapes.

    It is deliberately *not* a parent-domain suffix. A shared parent domain also carries app
    hostnames, per-cluster and per-tenant delegations, infrastructure and third-party services,
    and zones whose records are written elsewhere — while a single parent domain does not even
    cover every managed control plane. The rejections below are grouped by the shape rule that
    keeps each class out.
    """

    @pytest.mark.parametrize(
        "endpoint",
        [
            # Dedicated-hosted.
            "demo.hosted.unionai.cloud",
            "union-internal.hosted.unionai.cloud",
            # Regional dedicated.
            "acme.us-west-2.unionai.cloud",
            "acme.eu-west-1.unionai.cloud",
            "acme.eu-west-2.unionai.cloud",
            "acme.eu-central-1.unionai.cloud",
            # Serverless — a different parent domain, which a single suffix would miss.
            "acme.s.union.ai",
            "acme.us-east-2.s.union.ai",
            # Every accepted spelling of the same host.
            "dns:///acme.us-west-2.unionai.cloud",
            "https://demo.hosted.unionai.cloud:443",
            "https://acme.s.union.ai:443",
            "demo.hosted.unionai.cloud:443",
            "DEMO.Hosted.UnionAI.Cloud.",
            "dns:///ACME.S.Union.AI.",
        ],
    )
    def test_control_planes_are_allowed(self, endpoint):
        assert endpoint_allowed(endpoint) is True

    # -- must-reject: zones that are not control planes -------------------------------------

    @pytest.mark.parametrize(
        "endpoint",
        [
            "evil.sh-us-east-2.unionai.cloud",
            "x.sh-c-us-east-2.unionai.cloud",
            "cust.eks.unionai.cloud",
            "cust.gke.unionai.cloud",
            "svc.cicd-production.unionai.cloud",
        ],
    )
    def test_sibling_zones_are_rejected(self, endpoint):
        """Only the enumerated zone tails match.

        Sibling zones under the same parent host other things entirely, and some have their DNS
        records written outside this system — a name whose address someone else chooses is an
        SSRF primitive that a valid certificate does not make safe. They must be opted into
        explicitly rather than inherited from the parent domain.
        """
        assert endpoint_allowed(endpoint) is False

    def test_per_cluster_delegation_is_rejected(self):
        """A `<cluster>.dp.<org>.<zone>` delegation needs extra labels, so it cannot match."""
        assert endpoint_allowed("cluster1.dp.acme.hosted.unionai.cloud") is False

    @pytest.mark.parametrize(
        "endpoint",
        [
            "app.apps.acme.hosted.unionai.cloud",
            "flyte-mcp-central.apps.demo.hosted.unionai.cloud",
            "evil.apps.acme.us-west-2.unionai.cloud",
            "https://anything.apps.demo.hosted.unionai.cloud",
        ],
    )
    def test_deployed_apps_are_rejected(self, endpoint):
        """App hostnames serve user code; dialing one would hand it the forwarded token."""
        assert endpoint_allowed(endpoint) is False

    def test_internal_host_is_rejected(self):
        """`*.internal.*` names need extra labels, so they cannot match a tenant pattern."""
        assert endpoint_allowed("control-plane.internal.acme.sh-c-us-east-2.unionai.cloud") is False

    # -- must-reject: reserved labels and bare domains --------------------------------------

    def test_bare_infrastructure_host_is_rejected(self):
        """A host sitting directly under the parent domain is not a tenant shape."""
        assert endpoint_allowed("registry.unionai.cloud") is False

    @pytest.mark.parametrize("endpoint", ["signin.hosted.unionai.cloud", "auth.hosted.unionai.cloud"])
    def test_reserved_labels_are_rejected_in_tenant_position(self, endpoint):
        """Reserved labels fit the tenant shape but name services, so they are rejected."""
        assert endpoint_allowed(endpoint) is False

    @pytest.mark.parametrize("label", sorted(RESERVED_ORG_LABELS))
    def test_every_reserved_label_is_rejected_in_every_zone(self, label):
        for host in (
            f"{label}.hosted.unionai.cloud",
            f"{label}.us-west-2.unionai.cloud",
            f"{label}.s.union.ai",
            f"{label}.us-east-2.s.union.ai",
        ):
            assert endpoint_allowed(host) is False, host

    # -- must-reject: apexes, lookalikes, SSRF ---------------------------------------------

    @pytest.mark.parametrize("endpoint", ["unionai.cloud", "union.ai", "hosted.unionai.cloud", "s.union.ai"])
    def test_apex_domains_are_rejected(self, endpoint):
        """The apex is not a tenant, and a wildcard A record means it resolves anyway."""
        assert endpoint_allowed(endpoint) is False

    @pytest.mark.parametrize("endpoint", ["evil.com", "unionai.cloud.evil.com", "union.ai.evil.com"])
    def test_attacker_domains_are_rejected(self, endpoint):
        assert endpoint_allowed(endpoint) is False

    def test_fragment_spoof_is_rejected(self):
        """urlparse drops the fragment at connect time, so the real target is evil.com."""
        assert endpoint_allowed("evil.com#demo.hosted.unionai.cloud") is False

    @pytest.mark.parametrize("endpoint", ["169.254.169.254", "localhost", "127.0.0.1", "[::1]:8080"])
    def test_ip_literals_and_loopback_are_rejected(self, endpoint):
        """The cloud metadata service and loopback are the classic SSRF targets."""
        assert endpoint_allowed(endpoint) is False

    # -- shape of <org> ---------------------------------------------------------------------

    @pytest.mark.parametrize(
        "endpoint",
        [
            "a.b.hosted.unionai.cloud",  # two labels where one is allowed
            "-acme.hosted.unionai.cloud",  # leading hyphen is not a legal label
            "acme-.hosted.unionai.cloud",  # trailing hyphen likewise
            ".hosted.unionai.cloud",  # empty org
            "acme.us-east-1.unionai.cloud",  # region Union does not operate
            "acme.us-west-2.union.ai",  # right region, wrong zone
            "acme.s.unionai.cloud",  # serverless shape under the wrong parent
            "acme.us-east-2.union.ai",  # missing the `s` label
        ],
    )
    def test_org_must_be_one_label_in_a_known_zone(self, endpoint):
        assert endpoint_allowed(endpoint) is False

    def test_patterns_are_anchored(self):
        # A regex that only searched would match a suffix or prefix of an attacker's host.
        for pattern in DEFAULT_ALLOWED_ENDPOINT_PATTERNS:
            assert pattern.pattern.startswith("^")
            assert pattern.pattern.endswith("$")
