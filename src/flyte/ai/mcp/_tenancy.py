"""Multi-tenant request routing for a centrally hosted Flyte MCP server.

A per-tenant MCP deployment can bind one Flyte endpoint for the whole process because every
caller belongs to the same org. A *central* deployment (one hostname serving every Union
customer) cannot: the tenant is whatever the inbound credential says it is. This module
supplies the pieces that make that safe.

- `endpoint_allowed` — an allowlist so a forged credential cannot point the server at an
  attacker-controlled control plane (SSRF). See
  `DEFAULT_ALLOWED_ENDPOINT_PATTERNS` for why the default is an explicit set of
  control-plane host *shapes* and emphatically **not** a parent-domain suffix.
- `RateLimiter` — a per-credential token bucket so one runaway agent cannot spend the
  shared deployment's capacity on everybody else's behalf.
- `ClientCache` — one `flyte.remote._client.controlplane.ClientSet` per distinct
  credential, so a burst of requests from the same tenant doesn't re-run TLS + OAuth discovery
  on every call. It also validates the OAuth ``token_endpoint`` the target control plane
  advertises before the client-credentials flow can POST the tenant's secret to it.
- `CentralTenantMiddleware` — resolves the credential on each request and installs the
  matching init-config for the duration of the request via
  `flyte._initialize.init_config_context`, so ordinary ``flyte.remote`` calls inside the
  tool handlers transparently talk to the right tenant.

No credential is ever logged, and none is persisted server-side beyond the in-memory cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import logging
import math
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Sequence

if TYPE_CHECKING:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response

    from flyte._initialize import _InitConfig
else:
    try:
        from starlette.middleware.base import BaseHTTPMiddleware
    except ImportError:  # pragma: no cover - starlette is part of the ``mcp`` extra

        class BaseHTTPMiddleware:  # type: ignore[no-redef]
            pass


logger = logging.getLogger(__name__)

#: Env var holding a comma-separated list of endpoint suffixes the central server may talk to.
#: Providing it **replaces** `DEFAULT_ALLOWED_ENDPOINT_PATTERNS` with plain suffix /
#: exact-host matching — the escape hatch a self-hosted or private deployment uses to name its
#: own control planes.
ALLOWED_ENDPOINT_SUFFIXES_ENV_VAR = "FLYTE_MCP_ALLOWED_ENDPOINT_SUFFIXES"

#: One DNS label: what an org name is allowed to be. The single-label restriction is the whole
#: point of the default allowlist — see `DEFAULT_ALLOWED_ENDPOINT_PATTERNS`.
_ORG_LABEL = r"(?P<org>[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)"

#: Regions with a managed ``<org>.<region>.unionai.cloud`` control plane. Enumerated rather than
#: matched with a wildcard: sibling zones under the same parent are not control planes, and some
#: are not managed by this system at all. Adding a region here is a deliberate act.
_UNION_HOSTED_REGIONS: tuple[str, ...] = ("us-west-2", "eu-west-1", "eu-west-2", "eu-central-1")

#: Host shapes the central server may dial when no explicit allowlist is configured.
#:
#: **Why this is a pattern set and not a parent-domain suffix — do not "simplify" it back.** A
#: shared parent domain is not evidence that a host is a control plane. The same domain also
#: carries application hostnames, per-cluster and per-tenant delegations, infrastructure and
#: third-party services, and zones whose DNS records are writable outside this system. Matching a
#: suffix would accept all of them, and a name whose address is chosen by someone else is an SSRF
#: primitive that TLS verification does not catch. A suffix is also *incomplete*: not every
#: managed control plane lives under one parent domain, so it locks out legitimate tenants.
#:
#: ``<org>`` is therefore exactly **one** DNS label, and that is load-bearing: anything needing an
#: extra label (an app host, a per-cluster delegation, an internal name) cannot match, and the
#: fixed zone tails keep sibling zones out.
#:
#: Deployments whose control planes are not in this set — self-hosted, private, or otherwise —
#: name them explicitly via `ALLOWED_ENDPOINT_SUFFIXES_ENV_VAR`. That is an operator
#: deciding which zones to trust, rather than this server extending trust implicitly.
DEFAULT_ALLOWED_ENDPOINT_PATTERNS: tuple[re.Pattern[str], ...] = (
    # <org>.hosted.unionai.cloud
    re.compile(rf"^{_ORG_LABEL}\.hosted\.unionai\.cloud$"),
    # <org>.<region>.unionai.cloud, region enumerated (never a wildcard)
    re.compile(rf"^{_ORG_LABEL}\.(?:{'|'.join(re.escape(r) for r in _UNION_HOSTED_REGIONS)})\.unionai\.cloud$"),
    # <org>.s.union.ai — serverless
    re.compile(rf"^{_ORG_LABEL}\.s\.union\.ai$"),
    # <org>.us-east-2.s.union.ai — regional serverless
    re.compile(rf"^{_ORG_LABEL}\.us-east-2\.s\.union\.ai$"),
)

#: Single labels that fit a pattern above but never name a tenant — they are reserved for
#: infrastructure or third-party services. Rejected in the ``<org>`` slot even on a match.
RESERVED_ORG_LABELS: frozenset[str] = frozenset(
    {"signin", "auth", "registry", "connect", "tunnel", "argocd", "api", "bastion", "console", "uss"}
)

#: Env var holding extra suffixes accepted for a tenant's OAuth2 ``token_endpoint`` on top of
#: `ALLOWED_ENDPOINT_SUFFIXES_ENV_VAR`. The escape hatch for a tenant whose IdP legitimately
#: lives off the control-plane domain (e.g. an Okta tenant).
ALLOWED_TOKEN_ENDPOINT_SUFFIXES_ENV_VAR = "FLYTE_MCP_ALLOWED_TOKEN_ENDPOINT_SUFFIXES"

#: Env vars tuning the per-credential request throttle. ``0`` disables it.
RATE_LIMIT_RPM_ENV_VAR = "FLYTE_MCP_RATE_LIMIT_RPM"
RATE_LIMIT_BURST_ENV_VAR = "FLYTE_MCP_RATE_LIMIT_BURST"

DEFAULT_RATE_LIMIT_RPM = 120
DEFAULT_RATE_LIMIT_BURST = 30

#: Cap on distinct credentials tracked by the throttle at once. Same shape as the client cache:
#: idle buckets are expired, then the least recently used is dropped.
DEFAULT_MAX_RATE_LIMIT_BUCKETS = 4096

#: Label marking a hostname as a deployed Flyte *app* (``<name>.apps.<org>.…``). Apps are
#: customer-controlled HTTP servers that happen to live under the same parent domain as the
#: control planes, so they must never be accepted as one: dialing an app would hand it the
#: caller's forwarded token and make this server a proxy to arbitrary customer code.
_APP_HOST_LABEL = ".apps."

#: Header carrying the target endpoint when the caller presents a raw bearer token instead of
#: an API key (the API key already encodes its own endpoint).
ENDPOINT_HEADER = "x-union-endpoint"

#: Paths served without a credential.
#: Only liveness. A deployment that also wants an unauthenticated landing page must opt into
#: it explicitly — defaulting "/" open here would silently expose it at every other call site.
DEFAULT_EXCLUDED_PATHS: frozenset[str] = frozenset({"/health"})

DEFAULT_MAX_CACHE_ENTRIES = 512
DEFAULT_CACHE_TTL_S = 3600.0

#: How long a credential stays "proven" before the next request re-checks it against the
#: control plane. Short enough that a revoked key stops working promptly, long enough that a
#: busy agent session does not pay an identity round-trip per tool call.
VERIFY_TTL_S = 300.0

_API_KEY_HINT = (
    "Create an API key with `flyte create api-key --name mcp` (requires the `flyteplugins-union` "
    "package) and send it as `Authorization: Bearer <api-key>`."
)


# ------------------------------
# Endpoint allowlist
# ------------------------------


def configured_endpoint_suffixes() -> list[str] | None:
    """Return the operator-configured endpoint suffixes, or ``None`` when there are none.

    Reads `ALLOWED_ENDPOINT_SUFFIXES_ENV_VAR` (comma-separated). ``None`` means "no
    operator override", i.e. `DEFAULT_ALLOWED_ENDPOINT_PATTERNS` applies; a non-empty list
    *replaces* those patterns with suffix / exact-host matching.
    """
    raw = os.environ.get(ALLOWED_ENDPOINT_SUFFIXES_ENV_VAR)
    if not raw:
        return None
    items = [s.strip() for s in raw.split(",")]
    items = [s for s in items if s]
    return items or None


def endpoint_hostname(endpoint: str) -> str:
    """Reduce an endpoint in any of the accepted spellings to a bare lowercase hostname.

    Handles ``dns:///host``, ``https://host``, ``http://host:port`` and bare ``host:port``.
    Returns an empty string when nothing hostname-like can be extracted.
    """
    value = (endpoint or "").strip()
    if not value:
        return ""

    for prefix in ("dns:///", "dns://", "https://", "http://", "grpc://", "grpcs://"):
        if value.lower().startswith(prefix):
            value = value[len(prefix) :]
            break
    else:
        # A scheme we don't recognise is not something we should silently accept.
        if "://" in value:
            return ""

    # Drop any path / query / fragment the caller tacked on, then the port. All three must be
    # stripped before the suffix test: urlparse discards a fragment when the client later
    # connects, so `evil.com#ok.hosted.unionai.cloud` would otherwise pass the allowlist and
    # then dial evil.com.
    value = value.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    if value.startswith("["):  # bracketed IPv6 literal, with or without a port
        value = value[1:].split("]", 1)[0]
    else:
        host, sep, port = value.rpartition(":")
        if sep and port.isdigit():
            value = host
    host = value.strip().rstrip(".").lower()
    # Anything that is not a plain hostname (or IP literal) is rejected outright rather than
    # normalized, so no exotic spelling can reach the suffix comparison.
    if not _HOSTNAME_RE.match(host):
        return ""
    return host


#: A hostname the server is willing to dial: DNS labels, or a bare IPv4/IPv6 literal (which the
#: allowlist then rejects unless it was named exactly).
_HOSTNAME_RE = re.compile(r"^[a-z0-9._:-]+$")


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _is_local_or_literal(host: str) -> bool:
    """True for hosts that must never be reachable by pattern/suffix match, only by exact name."""
    return _is_ip_literal(host) or host == "localhost" or host.endswith(".localhost")


def _matches_default_patterns(host: str) -> bool:
    """Match ``host`` against `DEFAULT_ALLOWED_ENDPOINT_PATTERNS`."""
    if _is_local_or_literal(host):
        return False
    # Redundant against the single-label patterns (an app host carries extra labels), kept as a
    # second line of defence should a broader pattern ever be added.
    if _APP_HOST_LABEL in host:
        return False
    for pattern in DEFAULT_ALLOWED_ENDPOINT_PATTERNS:
        match = pattern.match(host)
        if match is not None and match.group("org") not in RESERVED_ORG_LABELS:
            return True
    return False


def _matches_suffixes(host: str, suffixes: Sequence[str]) -> bool:
    """Match ``host`` against an operator-supplied suffix / exact-host allowlist."""
    normalized = [s.strip().rstrip(".").lower() for s in suffixes]
    normalized = [s for s in normalized if s]

    # Exact allowlist entries win outright — that is the only way to permit localhost or an IP.
    if host in {s for s in normalized if not s.startswith(".")}:
        return True

    if _is_local_or_literal(host):
        return False

    if _APP_HOST_LABEL in host:
        return False

    for suffix in normalized:
        if suffix.startswith("."):
            if host.endswith(suffix):
                return True
        elif host.endswith(f".{suffix}"):
            return True
    return False


def endpoint_allowed(endpoint: str, suffixes: Sequence[str] | None = None) -> bool:
    """Return True when ``endpoint`` is one the central server is permitted to reach.

    Two modes, and which one runs depends on whether an operator configured an allowlist:

    - **Default (no ``suffixes``, no** `ALLOWED_ENDPOINT_SUFFIXES_ENV_VAR` **)** — the host
      must match one of `DEFAULT_ALLOWED_ENDPOINT_PATTERNS`, i.e. be a Union-operated
      control plane of the shape ``<org>.hosted.unionai.cloud``,
      ``<org>.<region>.unionai.cloud``, ``<org>.s.union.ai`` or ``<org>.us-east-2.s.union.ai``
      with ``<org>`` a single DNS label that is not one of `RESERVED_ORG_LABELS`. This is
      *not* a parent-domain suffix check, deliberately: read the note on
      `DEFAULT_ALLOWED_ENDPOINT_PATTERNS` before touching it.
    - **Configured** — ``suffixes`` (or the env var, which it mirrors) *replaces* those patterns.
      Entries beginning with a dot (``.example.com``) match any host under that domain; entries
      without one are an exact host allowlist (and also match subdomains of it). This is how a
      self-hosted or otherwise private deployment names its own control planes, which the
      defaults intentionally do not cover.

    In both modes IP literals and loopback names are rejected unless allowlisted *exactly*, so a
    credential naming ``169.254.169.254`` or ``localhost`` cannot turn the server into an SSRF
    proxy, and deployed-app hostnames (``<name>.apps.<org>.…``) are rejected outright: they sit
    under the tenant domain but are customer-controlled servers, not control planes.

    :param endpoint: Endpoint in any accepted spelling (``dns:///h``, ``https://h``, ``h:443``)
    :param suffixes: Explicit allowlist replacing the defaults; ``None`` consults
        `configured_endpoint_suffixes` and then the default patterns
    """
    host = endpoint_hostname(endpoint)
    if not host:
        return False

    allowed = list(suffixes) if suffixes is not None else configured_endpoint_suffixes()
    if allowed is None:
        return _matches_default_patterns(host)
    return _matches_suffixes(host, allowed)


def extra_allowed_token_endpoint_suffixes() -> list[str]:
    """Return the extra suffixes accepted for an OAuth ``token_endpoint``.

    Reads `ALLOWED_TOKEN_ENDPOINT_SUFFIXES_ENV_VAR` (comma-separated) and returns an
    empty list when unset — the endpoint allowlist alone is then the whole rule.
    """
    raw = os.environ.get(ALLOWED_TOKEN_ENDPOINT_SUFFIXES_ENV_VAR)
    if not raw:
        return []
    return [s for s in (item.strip() for item in raw.split(",")) if s]


class TokenEndpointNotAllowed(Exception):
    """A tenant's control plane advertised an OAuth token endpoint we refuse to POST to.

    Raised by `ClientCache`; `CentralTenantMiddleware` renders it as a 403.
    """


# ------------------------------
# Per-credential rate limiting
# ------------------------------


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        logger.warning(f"Ignoring non-integer {name}={raw!r}; using {default}.")
        return default
    if value < 0:
        logger.warning(f"Ignoring negative {name}={raw!r}; using {default}.")
        return default
    return value


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    """Token bucket keyed by a hashed credential.

    A central deployment is one shared endpoint in front of every tenant, so without a throttle
    a single customer's runaway agent spends the whole deployment's capacity and degrades
    everyone else. Each distinct credential gets its own bucket holding ``burst`` tokens that
    refill at ``rpm / 60`` per second: short bursts are absorbed, a sustained loop settles at
    ``rpm`` requests per minute.

    **Warning: the limit is per replica.** The central app runs with 4 replicas behind a load
    balancer and this bucket lives in process memory, so a caller spread across replicas sees
    an effective ceiling of ``rpm x replicas`` (480/min at the defaults). Treat it as a
    blast-radius cap on any one process, not as an exact global quota — that would need
    shared state (e.g. Redis).

    Buckets are held in an LRU capped at ``max_buckets``. A bucket that has refilled to capacity
    carries no state, so idle ones are dropped once they are older than the time a full refill
    takes; only under cap pressure can a still-throttled bucket be evicted early, and 4096
    concurrent credentials is far beyond the deployment's real fan-out.

    ``check`` is deliberately synchronous: it contains no ``await``, so the event loop cannot
    interleave two callers inside it and no lock is needed.

    :param rpm: Sustained requests per minute per credential. ``0`` disables the limiter.
        Defaults to `RATE_LIMIT_RPM_ENV_VAR`, then `DEFAULT_RATE_LIMIT_RPM`.
    :param burst: Bucket capacity — how many requests may arrive back-to-back. Defaults to
        `RATE_LIMIT_BURST_ENV_VAR`, then `DEFAULT_RATE_LIMIT_BURST`. ``0`` means
        "no separate burst allowance", i.e. capacity falls back to ``rpm``.
    :param max_buckets: Cap on tracked credentials.
    """

    def __init__(
        self,
        *,
        rpm: int | None = None,
        burst: int | None = None,
        max_buckets: int = DEFAULT_MAX_RATE_LIMIT_BUCKETS,
    ):
        self.rpm = _env_int(RATE_LIMIT_RPM_ENV_VAR, DEFAULT_RATE_LIMIT_RPM) if rpm is None else max(0, rpm)
        self.burst = _env_int(RATE_LIMIT_BURST_ENV_VAR, DEFAULT_RATE_LIMIT_BURST) if burst is None else max(0, burst)
        self._max_buckets = max(1, max_buckets)
        self._rate = self.rpm / 60.0
        self._capacity = float(self.burst or self.rpm)
        # A bucket only carries state while it is below capacity, and a full refill takes
        # exactly this long — so anything idle for longer is indistinguishable from a new one.
        self._idle_ttl_s = (self._capacity / self._rate) if self._rate > 0 else 0.0
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self.rpm > 0 and self._capacity > 0

    def __len__(self) -> int:
        return len(self._buckets)

    def check(self, key: str) -> int | None:
        """Consume one token for ``key``.

        :return: ``None`` when the request is allowed, else the whole number of seconds the
            caller should wait before retrying (suitable for a ``Retry-After`` header).
        """
        if not self.enabled:
            return None

        now = time.monotonic()
        self._expire(now)

        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=self._capacity, updated=now)
            self._buckets[key] = bucket
        else:
            bucket.tokens = min(self._capacity, bucket.tokens + (now - bucket.updated) * self._rate)
            bucket.updated = now
        self._buckets.move_to_end(key)
        while len(self._buckets) > self._max_buckets:
            self._buckets.popitem(last=False)

        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return None

        # Not enough for one request: report when the next whole token lands.
        return max(1, math.ceil((1.0 - bucket.tokens) / self._rate))

    def _expire(self, now: float) -> None:
        # Every touched key is moved to the end, so the dict is ordered by ``updated`` ascending
        # and the first live entry ends the sweep — this stays O(evicted), not O(tracked).
        while self._buckets:
            _, oldest = next(iter(self._buckets.items()))
            if now - oldest.updated <= self._idle_ttl_s:
                break
            self._buckets.popitem(last=False)


# ------------------------------
# Client cache
# ------------------------------


def _credential_key(credential: str) -> str:
    """Hash a credential so it can be used as a cache key without being held in plaintext."""
    return hashlib.sha256(credential.encode("utf-8")).hexdigest()


@dataclass
class _CacheEntry:
    cfg: _InitConfig
    last_used: float


async def _close_client(client: Any) -> None:
    """Best-effort release of a ClientSet's transport. Never raises.

    Note that today's ``ClientSet`` exposes no close/aclose (nor does its session config or
    underlying HTTP client), so eviction usually releases nothing and relies on GC. This stays
    as a hook for when the client grows one.
    """
    candidates = [client, getattr(client, "session_config", None)]
    session_config = getattr(client, "session_config", None)
    if session_config is not None:
        candidates.append(getattr(session_config, "http_client", None))

    for candidate in candidates:
        if candidate is None:
            continue
        for attr in ("aclose", "close"):
            closer = getattr(candidate, attr, None)
            if closer is None or not callable(closer):
                continue
            try:
                result = closer()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:  # pragma: no cover - purely defensive
                logger.debug(f"Ignoring error while closing cached client: {type(e).__name__}: {e}")
            return


class ClientCache:
    """Async-safe LRU of per-tenant `flyte._initialize._InitConfig` objects.

    Keys are ``sha256`` digests of the presenting credential, so the plaintext credential is
    never used as a dict key. Entries expire ``ttl_s`` after their last use and the least
    recently used entry is evicted once ``max_entries`` is exceeded; evicted clients get a
    best-effort transport close.

    Concurrent misses on the same key share one in-flight creation, so a burst of requests from
    a cold tenant produces exactly one ``ClientSet``.

    On the API-key path the cache also validates the OAuth ``token_endpoint`` the target control
    plane advertises — see `ClientCache._check_token_endpoint`.

    :param max_entries: LRU cap on cached configs
    :param ttl_s: Idle time after which a cached config is rebuilt
    :param root_dir: ``root_dir`` for the built ``_InitConfig``
    :param allowed_endpoint_suffixes: Suffix allowlist replacing
        `DEFAULT_ALLOWED_ENDPOINT_PATTERNS`, also used for the token-endpoint check;
        ``None`` uses the env var and then the default patterns
    :param validate_token_endpoint: Set False to skip the token-endpoint check entirely
    """

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_MAX_CACHE_ENTRIES,
        ttl_s: float = DEFAULT_CACHE_TTL_S,
        root_dir: Path | None = None,
        allowed_endpoint_suffixes: Sequence[str] | None = None,
        validate_token_endpoint: bool = True,
    ):
        self._max_entries = max_entries
        self._ttl_s = ttl_s
        self._root_dir = root_dir or Path.cwd()
        self._allowed_endpoint_suffixes = (
            list(allowed_endpoint_suffixes) if allowed_endpoint_suffixes is not None else None
        )
        self._validate_token_endpoint = validate_token_endpoint
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._pending: dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    def _token_endpoint_allowed(self, token_endpoint: str) -> bool:
        """Whether a control plane's advertised token endpoint may receive the tenant's secret.

        Accepted when the endpoint allowlist itself admits it (the ordinary case: a control plane
        hosting its own ``/oauth2/token``), or when it falls under the extra suffixes an operator
        listed in `ALLOWED_TOKEN_ENDPOINT_SUFFIXES_ENV_VAR`. The two are checked separately
        rather than concatenated so the extras work in default (pattern) mode too.
        """
        if endpoint_allowed(token_endpoint, self._allowed_endpoint_suffixes):
            return True
        extras = extra_allowed_token_endpoint_suffixes()
        return bool(extras) and endpoint_allowed(token_endpoint, extras)

    async def _check_token_endpoint(self, client: Any, *, endpoint: str) -> None:
        """Refuse to build a client whose control plane points the token POST somewhere else.

        Flyte's client-credentials flow asks the *target* endpoint for its OAuth2 metadata and
        then POSTs the tenant's client id + secret to whatever ``token_endpoint`` that response
        declares (``_token_client.get_token``). Nothing checks that URL, so an allowlisted but
        malicious (or compromised) control plane can aim the credential POST at any host on the
        internet — an outbound POST primitive running from inside the cluster. Resolving the
        config here, once per cache miss, closes that before the authenticator ever runs.

        The check is *central-mode only*: it lives in this module, not in
        ``flyte.remote._client.auth``, precisely because an ordinary user's external IdP (Okta,
        Auth0) legitimately hosts its token endpoint off the control-plane domain. Central
        deployments that need the same accommodate one tenant at a time via
        `ALLOWED_TOKEN_ENDPOINT_SUFFIXES_ENV_VAR`.

        :raises TokenEndpointNotAllowed: when the advertised token endpoint is off-allowlist
        """
        if not self._validate_token_endpoint:
            return

        from flyte.remote._client.auth import _client_config

        session_config = getattr(client, "session_config", None)
        address = getattr(session_config, "endpoint", None) or endpoint
        store = _client_config.RemoteClientConfigStore(
            address, http_client=getattr(session_config, "http_client", None)
        )
        client_config = await store.get_client_config()
        token_endpoint = client_config.token_endpoint or ""

        if not self._token_endpoint_allowed(token_endpoint):
            # The token endpoint is the control plane's own claim about itself, so echoing its
            # hostname leaks nothing of the caller's.
            raise TokenEndpointNotAllowed(
                f"Control plane '{endpoint_hostname(endpoint)}' advertised OAuth token endpoint "
                f"'{endpoint_hostname(token_endpoint) or token_endpoint}', which is not permitted by this "
                f"server's allowlist. Add its suffix to {ALLOWED_TOKEN_ENDPOINT_SUFFIXES_ENV_VAR} if it is "
                f"legitimate."
            )

    def __len__(self) -> int:
        return len(self._entries)

    def _build_config(self, client: Any, *, endpoint: str, org: str | None) -> _InitConfig:
        """Wrap a ``ClientSet`` in an ``_InitConfig`` carrying the tenant's org and endpoint.

        Both are load-bearing: ``.url`` properties on remote objects build console links off
        them, so a missing org sends the caller to the wrong tenant's console.
        """
        from flyte._initialize import _InitConfig

        return _InitConfig(
            root_dir=self._root_dir,
            org=org,
            # project/domain are deliberately unset: a central server has no single default,
            # so tools must take them per call.
            project=None,
            domain=None,
            client=client,
            image_builder="remote",
        )

    async def get_for_api_key(self, api_key: str, *, endpoint: str, org: str | None) -> _InitConfig:
        """Return (creating if needed) the config for an encoded API key.

        The ``ClientSet``'s client-credentials authenticator fetches and refreshes its own
        token, so nothing here does a manual token exchange — but it will POST this key's
        client id + secret to the endpoint the control plane advertises, so
        `ClientCache._check_token_endpoint` vets that target before the client is handed out.

        :raises TokenEndpointNotAllowed: when the control plane's token endpoint is off-allowlist
        """

        from flyte._utils import org_from_endpoint

        async def factory() -> _InitConfig:
            from flyte.remote._client.controlplane import ClientSet

            client = await ClientSet.for_api_key(api_key)
            await self._check_token_endpoint(client, endpoint=endpoint)
            return self._build_config(client, endpoint=endpoint, org=org or org_from_endpoint(endpoint))

        return await self._get_or_create(_credential_key(api_key), factory)

    async def get_for_endpoint(self, endpoint: str, *, org: str | None = None) -> _InitConfig:
        """Return (creating if needed) a passthrough-auth config for ``endpoint``.

        Used for raw bearer tokens: the client carries no credentials of its own and instead
        forwards whatever ``flyte.remote.auth_metadata`` holds for the current request. Keyed by
        endpoint rather than by token, since the client is token-independent.

        No token-endpoint check here, and none is needed: ``PassthroughAuthenticator`` skips the
        base ``Authenticator.__init__`` entirely — it holds no ``cfg_store``, never calls
        ``_resolve_config``, and never reaches ``_token_client.get_token``. The passthrough path
        does no token exchange at all, so there is no outbound credential POST to redirect.
        """
        from flyte._utils import org_from_endpoint, sanitize_endpoint

        normalized = sanitize_endpoint(endpoint) or endpoint

        async def factory() -> _InitConfig:
            from flyte.remote._client.controlplane import ClientSet

            client = await ClientSet.for_endpoint(normalized, auth_type="Passthrough")
            return self._build_config(client, endpoint=normalized, org=org or org_from_endpoint(normalized))

        return await self._get_or_create(f"endpoint:{_credential_key(normalized)}", factory)

    async def _get_or_create(self, key: str, factory: Callable[[], Awaitable[_InitConfig]]) -> _InitConfig:
        now = time.monotonic()
        async with self._lock:
            self._expire_locked(now)
            entry = self._entries.get(key)
            if entry is not None:
                entry.last_used = now
                self._entries.move_to_end(key)
                return entry.cfg

            pending = self._pending.get(key)
            owner = pending is None
            if pending is None:
                pending = asyncio.ensure_future(factory())
                self._pending[key] = pending

        try:
            # Shielded so a client that disconnects mid-handshake doesn't cancel the creation
            # that other concurrent requests for the same tenant are waiting on.
            cfg = await asyncio.shield(pending)
        finally:
            if owner:
                async with self._lock:
                    self._pending.pop(key, None)

        evicted: list[_InitConfig] = []
        async with self._lock:
            self._entries[key] = _CacheEntry(cfg=cfg, last_used=time.monotonic())
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                _, dropped = self._entries.popitem(last=False)
                evicted.append(dropped.cfg)

        for dropped_cfg in evicted:
            await _close_client(dropped_cfg.client)
        return cfg

    def _expire_locked(self, now: float) -> None:
        stale = [k for k, e in self._entries.items() if now - e.last_used > self._ttl_s]
        for k in stale:
            self._entries.pop(k, None)


# ------------------------------
# Middleware
# ------------------------------


def _json_error(status_code: int, detail: str, *, headers: dict[str, str] | None = None) -> Response:
    from starlette.responses import JSONResponse

    return JSONResponse(status_code=status_code, content={"detail": detail}, headers=headers)


class CentralTenantMiddleware(BaseHTTPMiddleware):
    """Resolve the tenant from each request's credential and scope Flyte calls to it.

    Two credential shapes are accepted on ``Authorization: Bearer <cred>``:

    1. An encoded Union **API key**, which already carries its own endpoint and org. The
       endpoint is checked against the allowlist and a client-credentials ``ClientSet`` is
       built (and cached) for it.
    2. A **raw bearer token** (e.g. one an already-authenticated client holds). Since a token
       says nothing about where it is valid, the caller must also send
       ``X-Union-Endpoint``; the request is then served by a passthrough-auth client that
       forwards the token verbatim.

    Either way the resolved config is installed with
    `flyte._initialize.init_config_context` for the duration of the request only, so
    concurrent requests from different tenants never observe each other's client.

    Every credential is additionally throttled by a `RateLimiter` before any client is
    built, so an unverified caller costs a dict lookup rather than a control-plane round-trip.

    :param app: The ASGI application to wrap
    :param allowed_endpoint_suffixes: Suffix allowlist replacing
        `DEFAULT_ALLOWED_ENDPOINT_PATTERNS`; ``None`` uses
        `ALLOWED_ENDPOINT_SUFFIXES_ENV_VAR` and then those patterns
    :param excluded_paths: Paths served without a credential (default ``/health`` and ``/``)
    :param cache: Client cache to use; a fresh `ClientCache` is created when omitted
    :param decoder: Override for ``decode_api_key`` (testing seam)
    :param verifier: Override for the credential-verification call (testing seam)
    :param rate_limiter: Throttle to use; a fresh env-configured `RateLimiter` when omitted
    """

    def __init__(
        self,
        app,
        allowed_endpoint_suffixes: Sequence[str] | None = None,
        excluded_paths: set[str] | None = None,
        cache: ClientCache | None = None,
        decoder: Callable[[str], tuple[str, str, str, str]] | None = None,
        verifier: Callable[[], Awaitable[Any]] | None = None,
        rate_limiter: RateLimiter | None = None,
    ):
        super().__init__(app)
        self.allowed_endpoint_suffixes = (
            list(allowed_endpoint_suffixes) if allowed_endpoint_suffixes is not None else None
        )
        self.excluded_paths = set(DEFAULT_EXCLUDED_PATHS if excluded_paths is None else excluded_paths)
        self.cache = (
            cache if cache is not None else ClientCache(allowed_endpoint_suffixes=self.allowed_endpoint_suffixes)
        )
        self.rate_limiter = rate_limiter if rate_limiter is not None else RateLimiter()
        self._decoder = decoder
        self._verifier = verifier
        # Credentials proven good against their control plane, keyed like the client cache.
        self._verified: OrderedDict[str, float] = OrderedDict()
        self._verify_lock = asyncio.Lock()

    async def _verify_credential(self, key: str) -> bool:
        """Prove the caller's credential against its control plane before serving anything.

        Decoding an API key is pure string work and building a ``ClientSet`` does no I/O, so
        without this a forged credential would reach the tools — every request must cost one
        identity round-trip the first time, then ride the cache.
        """
        now = time.monotonic()
        async with self._verify_lock:
            seen_at = self._verified.get(key)
            if seen_at is not None and now - seen_at < VERIFY_TTL_S:
                self._verified.move_to_end(key)
                return True

        try:
            if self._verifier is not None:
                await self._verifier()
            else:
                import flyte.remote

                await flyte.remote.User.get.aio()
        except Exception as e:
            logger.info(f"Credential verification failed: {type(e).__name__}")
            return False

        async with self._verify_lock:
            self._verified[key] = now
            self._verified.move_to_end(key)
            while len(self._verified) > DEFAULT_MAX_CACHE_ENTRIES:
                self._verified.popitem(last=False)
        return True

    def _decode_api_key(self, credential: str) -> tuple[str, str, str, str]:
        if self._decoder is not None:
            return self._decoder(credential)
        from flyte.remote._client.auth._auth_utils import decode_api_key

        return decode_api_key(credential)  # type: ignore[return-value]

    def _endpoint_allowed(self, endpoint: str) -> bool:
        return endpoint_allowed(endpoint, self.allowed_endpoint_suffixes)

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        credential = _bearer_credential(request.headers.get("authorization"))
        if credential is None:
            return _json_error(
                401,
                f"Authentication required. {_API_KEY_HINT}",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Throttle on the credential alone, before any ClientSet is built or verified: an
        # unverified caller must stay cheap to reject.
        credential_key = _credential_key(credential)
        retry_after = self.rate_limiter.check(credential_key)
        if retry_after is not None:
            return _json_error(
                429,
                (
                    f"Rate limit exceeded for this credential: "
                    f"{self.rate_limiter.rpm} requests/minute (burst {self.rate_limiter.burst}). "
                    f"Retry in {retry_after}s."
                ),
                headers={"Retry-After": str(retry_after)},
            )

        try:
            endpoint, _client_id, _client_secret, org = self._decode_api_key(credential)
            is_api_key = bool(endpoint)
        except Exception:
            # Not an API key (not base64, or not the 4-part payload). Fall through to the raw
            # bearer path. The exception text can embed the credential, so it is never logged.
            is_api_key = False
            endpoint, org = "", "None"

        if not is_api_key:
            return await self._dispatch_bearer(request, call_next, credential)

        host = endpoint_hostname(endpoint)
        if not self._endpoint_allowed(endpoint):
            return _json_error(403, f"Endpoint '{host}' is not permitted by this server's endpoint allowlist.")

        normalized_org = None if org in (None, "", "None") else org
        try:
            cfg = await self.cache.get_for_api_key(credential, endpoint=endpoint, org=normalized_org)
        except TokenEndpointNotAllowed as e:
            # A policy rejection, not a transport failure: the message names only hostnames the
            # control plane published about itself, so it is safe to return verbatim.
            logger.warning(f"Rejected token endpoint for {host}: {e}")
            return _json_error(403, str(e))
        except Exception as e:
            logger.warning(f"Failed to build client for endpoint {host}: {type(e).__name__}")
            return _json_error(502, f"Could not establish a session with '{host}'.")

        from flyte._initialize import init_config_context

        with init_config_context(cfg):
            if not await self._verify_credential(credential_key):
                return _json_error(
                    401,
                    f"Credential was rejected by '{host}'. {_API_KEY_HINT}",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            logger.info("MCP central request org=%r endpoint=%r path=%r", normalized_org, host, request.url.path)
            return await call_next(request)

    async def _dispatch_bearer(self, request: Request, call_next, credential: str) -> Response:
        endpoint = request.headers.get(ENDPOINT_HEADER)
        if not endpoint:
            return _json_error(
                401,
                (
                    f"Credential is not a Union API key; send the target control plane in the "
                    f"'{ENDPOINT_HEADER}' header to use a raw bearer token. {_API_KEY_HINT}"
                ),
                headers={"WWW-Authenticate": "Bearer"},
            )

        host = endpoint_hostname(endpoint)
        if not self._endpoint_allowed(endpoint):
            return _json_error(403, f"Endpoint '{host}' is not permitted by this server's endpoint allowlist.")

        try:
            cfg = await self.cache.get_for_endpoint(endpoint)
        except Exception as e:
            logger.warning(f"Failed to build passthrough client for endpoint {host}: {type(e).__name__}")
            return _json_error(502, f"Could not establish a session with '{host}'.")

        from flyte._initialize import init_config_context
        from flyte.remote import auth_metadata

        with init_config_context(cfg), auth_metadata(("authorization", f"Bearer {credential}")):
            if not await self._verify_credential(f"bearer:{_credential_key(credential)}:{host}"):
                return _json_error(
                    401,
                    f"Token was rejected by '{host}'. {_API_KEY_HINT}",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            logger.info("MCP central request org=%r endpoint=%r path=%r auth=bearer", cfg.org, host, request.url.path)
            return await call_next(request)


def _bearer_credential(header_value: str | None) -> str | None:
    """Extract the credential from an ``Authorization: Bearer <cred>`` header value."""
    if not header_value:
        return None
    scheme, _, rest = header_value.partition(" ")
    if scheme.lower() != "bearer":
        return None
    credential = rest.strip()
    return credential or None
