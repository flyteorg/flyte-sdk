"""OAuth 2.0 Protected Resource Metadata ([RFC 9728](https://www.rfc-editor.org/rfc/rfc9728)) for MCP servers.

MCP clients that implement the MCP authorization spec find out how to sign in from two things
the server provides: a `401` whose `WWW-Authenticate: Bearer resource_metadata="..."` header
points at a metadata document, and that document, which names the authorization server(s) that
issue tokens for this server. This module holds the user-facing configuration and the two
pieces of HTTP plumbing; `FlyteMCPAppEnvironment` wires them in.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable
from urllib.parse import unquote, urlsplit, urlunsplit

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.types import ASGIApp, Receive, Scope, Send

#: RFC 9728 §3: the well-known URI suffix for protected resource metadata.
WELL_KNOWN_PATH = "/.well-known/oauth-protected-resource"

#: RFC 6749 §3.3 scope token: printable ASCII except space, `"` and `\`.
_SCOPE_TOKEN = re.compile(r"[\x21\x23-\x5b\x5d-\x7e]+")

# These URLs are embedded in a `WWW-Authenticate` challenge, whose parameters are
# comma-separated. Quoting is not enough: the reference MCP client finds parameters with an
# unanchored regex, so a Host of `a.example,scope=admin` puts a readable `scope` inside the
# quoted URL, and naive clients split on commas without honouring quotes. Characters are
# therefore allowlisted per component rather than denylisted.
_HOST_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
_AUTHORITY = re.compile(
    rf"(?:(?P<host>{_HOST_LABEL}(?:\.{_HOST_LABEL})*)|\[(?P<ipv6>[0-9A-Fa-f:.]+)\])(?::(?P<port>[0-9]{{1,5}}))?"
)
# RFC 3986 pchar and "/", without ",". ";" stays: path parameters are part of some resource
# paths, and the challenge value is quoted. Braces are outside RFC 3986 but were always
# accepted for issuer and documentation paths; `_require_resource_url` rejects them where
# they cannot be routed.
_PATH = re.compile(r"(?:[A-Za-z0-9\-._~!$&'()*+;=:@/{}]|%[0-9A-Fa-f]{2})*")
# The query additionally allows "?" and also excludes ";", which some parsers treat as a
# parameter separator.
_QUERY = re.compile(r"(?:[A-Za-z0-9\-._~!$&'()*+=:@/?{}]|%[0-9A-Fa-f]{2})*")
# The reference MCP client reads each challenge parameter with an unanchored
# `re.search(rf"{name}=...")`, so one of these names anywhere in the quoted resource_metadata
# URL, even as the tail of `xscope=`, is read as that parameter. They are rejected in any URL
# that reaches the challenge, matched case-insensitively as auth-param names are.
_CHALLENGE_PARAMETER_NAMES = ("scope=", "resource_metadata=", "error=", "error_description=", "error_uri=", "realm=")


def _require_authority(field_name: str, authority: str) -> None:
    """Accept only `host[:port]`: dot-separated DNS labels or a bracketed IPv6 literal.

    A trailing dot (`example.com.`) is rejected: it names the same host under a second
    spelling, and clients compare the advertised resource with their own URL as a string.
    """
    match = _AUTHORITY.fullmatch(authority)
    valid = match is not None and (match["port"] is None or 1 <= int(match["port"]) <= 65535)
    try:
        if match is not None and valid and match["ipv6"] is not None:
            ipaddress.IPv6Address(match["ipv6"])
        elif match is not None and valid:
            # RFC 1035 length limits. A final all-digit label cannot be a DNS name (RFC 3696
            # §2), so such a host must be a well-formed IPv4 address rather than `999.1.1.1`.
            labels = match["host"].split(".")
            valid = len(match["host"]) <= 253 and all(len(label) <= 63 for label in labels)
            if valid and labels[-1].isdigit():
                ipaddress.IPv4Address(match["host"])
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(
            f"{field_name} must be an absolute http(s) URL whose host is DNS labels or a bracketed "
            "IPv6 address, with an optional numeric port."
        )


def _require_http_url(field_name: str, url: str) -> None:
    try:
        parsed = urlsplit(url)
    except ValueError as e:
        raise ValueError(f"{field_name} must be an absolute http(s) URL.") from e
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"{field_name} must be an absolute http(s) URL.")
    if "@" in parsed.netloc or "#" in url:
        raise ValueError(f"{field_name} must be an absolute http(s) URL without userinfo or fragments.")
    _require_authority(field_name, parsed.netloc)
    # urlsplit silently deletes tab, CR and LF and strips leading control characters, so the
    # components must also add back up to the original string.
    query = f"?{parsed.query}" if parsed.query else ""
    if (
        f"{parsed.scheme}://{parsed.netloc}{parsed.path}{query}" != url
        or _PATH.fullmatch(parsed.path) is None
        or _QUERY.fullmatch(parsed.query) is None
    ):
        raise ValueError(
            f"{field_name} must be an absolute http(s) URL with unsafe characters (including , in the "
            "URL and ; in the query) percent-encoded."
        )


def _require_challenge_safe(field_name: str, text: str) -> None:
    lowered = text.lower()
    if any(name in lowered for name in _CHALLENGE_PARAMETER_NAMES):
        raise ValueError(
            f"{field_name} must not contain {', '.join(_CHALLENGE_PARAMETER_NAMES)} in any letter case: "
            "MCP clients would read them as WWW-Authenticate parameters. Percent-encode the '='."
        )


def _require_resource_path(field_name: str, path: str) -> None:
    """Check a path that is served as, and advertised in the challenge for, the resource."""
    if _PATH.fullmatch(path) is None:
        raise ValueError(f"{field_name} must be a URL path with unsafe characters (including ,) percent-encoded.")
    _require_challenge_safe(field_name, path)
    # ASGI decodes once: auth middleware reparses ?/# as URL delimiters, while
    # Starlette interprets braces as route parameters rather than literal path text.
    if any(c in unquote(path) for c in "?#{}"):
        raise ValueError(
            f"{field_name} path cannot be routed: after one percent-decoding it must not contain ?, #, {{ or }}. "
            "Choose a routable path."
        )


def _require_resource_url(field_name: str, url: str) -> None:
    _require_http_url(field_name, url)
    _require_challenge_safe(field_name, url)
    _require_resource_path(field_name, urlsplit(url).path)


def _require_issuer_url(url: str) -> None:
    _require_http_url("authorization_servers", url)
    parsed = urlsplit(url)
    if parsed.scheme != "https" or "?" in url or "#" in url:
        raise ValueError("authorization_servers must use HTTPS without a query or fragment.")


@dataclass(frozen=True, kw_only=True)
class ProtectedResourceMetadata:
    """What an MCP server advertises about its own authorization, per RFC 9728.

    Pass one to `FlyteMCPAppEnvironment.protected_resource_metadata` to let MCP clients that
    implement the [MCP authorization spec](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization)
    discover how to obtain a token instead of needing one pasted into their configuration.

    The identity provider must support authorization-code flow with PKCE S256 and either
    permit dynamic client registration, accept an administrator-approved client ID metadata
    document, or provide a preregistered client supported by the MCP client. Register the
    client's exact redirect URI, assign its users, and allow the requested scopes. Advertising
    a registration endpoint or metadata-document support does not grant permission to use it.

    MCP clients send this server's URL as the RFC 8707 `resource` parameter. The provider
    must accept that resource and issue an access token whose audience Flyte accepts; this
    metadata cannot change provider policy or Flyte's audience validation. Providers that
    reject anonymous registration require administrator setup before a complete sign-in flow.

    This feature retains token passthrough: Flyte validates tokens on remote tool calls,
    while local search tools accept any nonempty bearer token. It does not implement local
    audience validation or token exchange and does not establish full MCP OAuth security
    conformance. Configure those separately rather than weakening downstream token checks.

    Hostnames, configured or taken from the request's Host header, are limited to DNS labels
    (or a dotted IPv4 address) or a bracketed IPv6 address with an optional numeric port. URLs
    containing `,` (or `;` in the authority) are rejected because challenge parameters are
    comma-separated and some clients, including the reference MCP client, also parse unquoted
    values. That client also finds parameters anywhere in the header, so the resource URL and
    `mcp_mount_path` must not contain `scope=`, `resource_metadata=`, `error=`,
    `error_description=`, `error_uri=` or `realm=` in any letter case.

    Args:
        authorization_servers: Issuer URLs of the OAuth 2.0 authorization servers whose tokens
            this server accepts. `None` (the default) advertises the issuer that the Flyte
            endpoint the server is connected to advertises itself, fetched once when the
            metadata is first requested; the server forwards each caller's token to that
            endpoint, so that issuer is normally the right one. Issuers must use HTTPS
            without query, fragment or userinfo. Lists are copied at construction and
            revalidated before publication if changed in process.
        resource: The resource identifier, i.e. the full URL clients use to reach the MCP
            endpoint (for example `https://mcp.example.com/flyte-mcp/mcp`). Clients reject
            metadata whose `resource` does not match the URL they connected to. `None` (the
            default) uses the request's ASGI scheme and Host, plus the MCP endpoint path.
            INTERNAL_APP_ENDPOINT_PATTERN is a private service URL and is not used. Set
            resource explicitly when a proxy rewrites the public host or path. For TLS
            termination, configure Uvicorn's proxy_headers and forwarded_allow_ips (or
            FORWARDED_ALLOW_IPS) to trust only the proxy that supplies X-Forwarded-Proto.
            Raw Forwarded and X-Forwarded-Host headers are never used for discovery.
            Explicit resource paths must not contain ?, #, { or } after
            one percent-decoding: the current auth middleware and Starlette routing cannot
            serve their metadata paths. This limitation applies to the path, not a URL query.
        scopes_supported: OAuth scopes a client should request. Also sent as the `scope`
            parameter of the `401` challenge.
        resource_name: Human-readable name for the server, shown by some clients on their
            consent screen.
        resource_documentation: URL of documentation for developers using the server.

    Example:
    ```python
    from flyte.ai.mcp import FlyteMCPAppEnvironment, ProtectedResourceMetadata

    env = FlyteMCPAppEnvironment(
        name="flyte-mcp",
        requires_auth=False,
        protected_resource_metadata=ProtectedResourceMetadata(
            authorization_servers=["https://auth.example.com/oauth2/default"],
        ),
    )
    ```
    """

    authorization_servers: list[str] | None = None
    resource: str | None = None
    scopes_supported: list[str] | None = None
    resource_name: str | None = None
    resource_documentation: str | None = None

    def __post_init__(self):
        # A frozen dataclass does not own the caller's lists; snapshot them so later caller
        # mutations cannot bypass URL/scope validation.
        for name in ("authorization_servers", "scopes_supported"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, list(value))
        if self.authorization_servers is not None:
            if not self.authorization_servers:
                # RFC 9728 makes the list optional, but MCP requires at least one entry, and an
                # empty list is far more likely a bug than a request for discovery.
                raise ValueError(
                    "authorization_servers must not be empty; pass None to advertise the issuer "
                    "the Flyte endpoint advertises."
                )
            for url in self.authorization_servers:
                _require_issuer_url(url)
        if self.resource is not None:
            _require_resource_url("resource", self.resource)
        if self.resource_documentation is not None:
            _require_http_url("resource_documentation", self.resource_documentation)
        if self.scopes_supported is not None:
            for scope in self.scopes_supported:
                # RFC 6749 §3.3 excludes spaces, quotes and backslashes from scope tokens;
                # accepting them here would also let configuration inject challenge parameters.
                if _SCOPE_TOKEN.fullmatch(scope) is None:
                    raise ValueError("scopes_supported must contain valid OAuth scope tokens.")


def build_metadata_path(resource_path: str) -> str:
    """Return the path at which RFC 9728 §3.1 places the metadata for a resource with this path.

    The well-known suffix goes between the host and the resource's path, so the metadata for
    `https://host/flyte-mcp/mcp` lives at `https://host/.well-known/oauth-protected-resource/flyte-mcp/mcp`.
    This is the rule `mcp.server.auth.routes.build_resource_metadata_url` implements; it is
    restated here on a bare path because when `resource` is derived per request the host is
    not known when the route is registered, and the route and the `WWW-Authenticate` URL must
    come from the same function so they cannot disagree.
    """
    if resource_path in ("", "/"):
        return WELL_KNOWN_PATH
    return WELL_KNOWN_PATH + resource_path


class BearerChallengeMiddleware:
    """Answer requests that carry no bearer token with an RFC 9728 `401` challenge.

    `mcp.server.auth.middleware.bearer_auth.RequireAuthMiddleware` sends the same header, but
    only after a `TokenVerifier` has validated the token in-process. This server does not
    validate tokens itself — it forwards them to the Flyte endpoint on every remote call — so
    using it would mean stacking `AuthenticationMiddleware` and a verifier that accepts
    anything, just to test whether the header is present. This does that test directly.

    Args:
        app: The ASGI app to protect.
        resource_url: Returns the resource identifier for a request.
        excluded_paths: Paths served without a token (health check, the metadata itself).
        scopes: Sent as the challenge's `scope` parameter when given.
    """

    def __init__(
        self,
        app: ASGIApp,
        resource_url: Callable[[Request], str],
        excluded_paths: set[str],
        scopes: list[str] | None = None,
    ):
        self.app = app
        self.resource_url = resource_url
        self.excluded_paths = excluded_paths
        self.scopes = tuple(scopes or ())
        for scope in self.scopes:
            if _SCOPE_TOKEN.fullmatch(scope) is None:
                raise ValueError("scopes must contain valid OAuth scope tokens.")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        from starlette.requests import Request
        from starlette.responses import JSONResponse

        if scope["type"] != "http" or scope["path"] in self.excluded_paths:
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        auth_scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if auth_scheme.lower() == "bearer" and token.strip():
            await self.app(scope, receive, send)
            return

        try:
            resource_url = self.resource_url(request)
            _require_resource_url("resource", resource_url)
        except ValueError:
            await JSONResponse(status_code=400, content={"detail": "Invalid resource URL"})(scope, receive, send)
            return
        resource = urlsplit(resource_url)
        metadata_url = urlunsplit(
            (resource.scheme, resource.netloc, build_metadata_path(resource.path), resource.query, "")
        )
        # No `error` parameter: RFC 6750 §3.1 says to omit it when the request carried no
        # credentials at all, as opposed to carrying a bad one.
        challenge = f'Bearer resource_metadata="{metadata_url}"'
        if self.scopes:
            challenge += f', scope="{" ".join(self.scopes)}"'
        response = JSONResponse(
            status_code=401,
            # Same body as `FastAPIPassthroughAuthMiddleware`'s 401, which this replaces.
            content={"detail": "Authentication credentials required"},
            headers={"WWW-Authenticate": challenge},
        )
        await response(scope, receive, send)


class _AuthorizationServerDiscoveryError(RuntimeError):
    """Expected discovery failures; plain RuntimeError also covers programming errors."""


async def discover_authorization_server() -> str:
    """Return the OAuth issuer advertised by the Flyte endpoint the SDK is initialized against.

    This is the same `AuthMetadataService` lookup the SDK's own authenticators use to find
    where to sign in. The service is unauthenticated, so the call carries no credentials.

    Raises:
        _AuthorizationServerDiscoveryError: if the SDK is not initialized or the endpoint advertises no issuer.
    """
    from flyteidl2.auth.auth_service_connect import AuthMetadataServiceClient
    from flyteidl2.auth.auth_service_pb2 import GetOAuth2MetadataRequest

    from flyte._initialize import _get_init_config

    cfg = _get_init_config()
    if cfg is None or cfg.client is None:
        raise _AuthorizationServerDiscoveryError(
            "Flyte is not initialized, so its authorization server cannot be discovered. "
            "Set ProtectedResourceMetadata.authorization_servers explicitly."
        )
    session = cfg.client.session_config
    # The session's http client, but not its interceptors: those add the caller's credentials,
    # and in passthrough mode there is no caller here.
    client = AuthMetadataServiceClient(address=session.endpoint, http_client=session.http_client)
    metadata = await client.get_o_auth2_metadata(GetOAuth2MetadataRequest())
    if not metadata.issuer:
        raise _AuthorizationServerDiscoveryError(
            f"The Flyte endpoint {session.endpoint} advertises no OAuth issuer. "
            "Set ProtectedResourceMetadata.authorization_servers explicitly."
        )
    return metadata.issuer
