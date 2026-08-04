"""Multi-tenant request routing for a centrally hosted Flyte MCP server.

A per-tenant MCP deployment can bind one Flyte endpoint for the whole process because every
caller belongs to the same org. A *central* deployment (one hostname serving every Union
customer) cannot: the tenant is whatever the inbound credential says it is. This module
supplies the three pieces that makes that safe.

- :func:`endpoint_allowed` — an allowlist so a forged credential cannot point the server at an
  attacker-controlled control plane (SSRF).
- :class:`ClientCache` — one :class:`~flyte.remote._client.controlplane.ClientSet` per distinct
  credential, so a burst of requests from the same tenant doesn't re-run TLS + OAuth discovery
  on every call.
- :class:`CentralTenantMiddleware` — resolves the credential on each request and installs the
  matching init-config for the duration of the request via
  :func:`flyte._initialize.init_config_context`, so ordinary ``flyte.remote`` calls inside the
  tool handlers transparently talk to the right tenant.

No credential is ever logged, and none is persisted server-side beyond the in-memory cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import logging
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
ALLOWED_ENDPOINT_SUFFIXES_ENV_VAR = "FLYTE_MCP_ALLOWED_ENDPOINT_SUFFIXES"

#: Suffixes used when :data:`ALLOWED_ENDPOINT_SUFFIXES_ENV_VAR` is unset.
DEFAULT_ALLOWED_ENDPOINT_SUFFIXES: tuple[str, ...] = (".hosted.unionai.cloud",)

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


def default_allowed_endpoint_suffixes() -> list[str]:
    """Return the configured endpoint allowlist.

    Reads :data:`ALLOWED_ENDPOINT_SUFFIXES_ENV_VAR` (comma-separated) and falls back to
    :data:`DEFAULT_ALLOWED_ENDPOINT_SUFFIXES`.
    """
    raw = os.environ.get(ALLOWED_ENDPOINT_SUFFIXES_ENV_VAR)
    if not raw:
        return list(DEFAULT_ALLOWED_ENDPOINT_SUFFIXES)
    items = [s.strip() for s in raw.split(",")]
    items = [s for s in items if s]
    return items or list(DEFAULT_ALLOWED_ENDPOINT_SUFFIXES)


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


def endpoint_allowed(endpoint: str, suffixes: Sequence[str] | None = None) -> bool:
    """Return True when ``endpoint`` is one the central server is permitted to reach.

    ``suffixes`` entries beginning with a dot (``.hosted.unionai.cloud``) match any host under
    that domain; entries without one are treated as an exact host allowlist (and also match
    subdomains of it). IP literals and loopback names are rejected unless allowlisted exactly,
    so a credential naming ``169.254.169.254`` or ``localhost`` cannot turn the server into an
    SSRF proxy.

    :param endpoint: Endpoint in any accepted spelling (``dns:///h``, ``https://h``, ``h:443``)
    :param suffixes: Allowlist; defaults to :func:`default_allowed_endpoint_suffixes`
    """
    allowed = list(default_allowed_endpoint_suffixes() if suffixes is None else suffixes)
    host = endpoint_hostname(endpoint)
    if not host:
        return False

    normalized = [s.strip().rstrip(".").lower() for s in allowed]
    normalized = [s for s in normalized if s]

    # Exact allowlist entries win outright — that is the only way to permit localhost or an IP.
    if host in {s for s in normalized if not s.startswith(".")}:
        return True

    if _is_ip_literal(host) or host == "localhost" or host.endswith(".localhost"):
        return False

    for suffix in normalized:
        if suffix.startswith("."):
            if host.endswith(suffix):
                return True
        elif host.endswith(f".{suffix}"):
            return True
    return False


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
    """Async-safe LRU of per-tenant :class:`~flyte._initialize._InitConfig` objects.

    Keys are ``sha256`` digests of the presenting credential, so the plaintext credential is
    never used as a dict key. Entries expire ``ttl_s`` after their last use and the least
    recently used entry is evicted once ``max_entries`` is exceeded; evicted clients get a
    best-effort transport close.

    Concurrent misses on the same key share one in-flight creation, so a burst of requests from
    a cold tenant produces exactly one ``ClientSet``.
    """

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_MAX_CACHE_ENTRIES,
        ttl_s: float = DEFAULT_CACHE_TTL_S,
        root_dir: Path | None = None,
    ):
        self._max_entries = max_entries
        self._ttl_s = ttl_s
        self._root_dir = root_dir or Path.cwd()
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._pending: dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

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
        token, so nothing here does a manual token exchange.
        """

        from flyte._utils import org_from_endpoint

        async def factory() -> _InitConfig:
            from flyte.remote._client.controlplane import ClientSet

            client = await ClientSet.for_api_key(api_key)
            return self._build_config(client, endpoint=endpoint, org=org or org_from_endpoint(endpoint))

        return await self._get_or_create(_credential_key(api_key), factory)

    async def get_for_endpoint(self, endpoint: str, *, org: str | None = None) -> _InitConfig:
        """Return (creating if needed) a passthrough-auth config for ``endpoint``.

        Used for raw bearer tokens: the client carries no credentials of its own and instead
        forwards whatever ``flyte.remote.auth_metadata`` holds for the current request. Keyed by
        endpoint rather than by token, since the client is token-independent.
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
    :func:`~flyte._initialize.init_config_context` for the duration of the request only, so
    concurrent requests from different tenants never observe each other's client.

    :param app: The ASGI application to wrap
    :param allowed_endpoint_suffixes: Endpoint allowlist; defaults to the env-configured one
    :param excluded_paths: Paths served without a credential (default ``/health`` and ``/``)
    :param cache: Client cache to use; a fresh :class:`ClientCache` is created when omitted
    :param decoder: Override for ``decode_api_key`` (testing seam)
    """

    def __init__(
        self,
        app,
        allowed_endpoint_suffixes: Sequence[str] | None = None,
        excluded_paths: set[str] | None = None,
        cache: ClientCache | None = None,
        decoder: Callable[[str], tuple[str, str, str, str]] | None = None,
        verifier: Callable[[], Awaitable[Any]] | None = None,
    ):
        super().__init__(app)
        self.allowed_endpoint_suffixes = (
            list(allowed_endpoint_suffixes) if allowed_endpoint_suffixes is not None else None
        )
        self.excluded_paths = set(DEFAULT_EXCLUDED_PATHS if excluded_paths is None else excluded_paths)
        self.cache = cache if cache is not None else ClientCache()
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
        except Exception as e:
            logger.warning(f"Failed to build client for endpoint {host}: {type(e).__name__}")
            return _json_error(502, f"Could not establish a session with '{host}'.")

        from flyte._initialize import init_config_context

        with init_config_context(cfg):
            if not await self._verify_credential(_credential_key(credential)):
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
