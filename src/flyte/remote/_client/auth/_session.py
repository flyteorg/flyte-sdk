import asyncio
import socket
import typing
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import pyqwest
from OpenSSL import SSL, crypto

from flyte._logging import logger
from flyte._utils.org_discovery import hostname_from_url

from ._authenticators.base import get_async_session
from ._authenticators.factory import (
    create_auth_interceptors,
    create_proxy_auth_interceptors,
    get_async_proxy_authenticator,
)


@dataclass(frozen=True)
class SessionConfig:
    endpoint: str
    insecure: bool
    insecure_skip_verify: bool
    interceptors: tuple
    http_client: Any
    api_key: typing.Optional[str] = None

    def connect_kwargs(self) -> dict[str, Any]:
        return {"address": self.endpoint, "interceptors": self.interceptors, "http_client": self.http_client}


def normalize_rpc_endpoint(endpoint: str, *, insecure: bool = False) -> str:
    """Translate gRPC-style endpoint to http(s) URL for ConnectRPC."""
    scheme = "http" if insecure else "https"
    parsed = urlparse(endpoint)

    if parsed.scheme in ("http", "https"):
        return endpoint

    if parsed.scheme == "dns":
        host = parsed.path.lstrip("/")
        return f"{scheme}://{host}"

    # urlparse("example.com:8089") mis-parses "example.com" as the scheme and
    # leaves netloc empty.  A genuine URL like "ftp://example.com" will have a
    # non-empty netloc.  Use that to tell the two cases apart.
    if parsed.netloc:
        # A real URL with an unrecognised scheme (e.g. ftp://).
        raise ValueError(
            f"Unknown scheme '{parsed.scheme}' in endpoint '{endpoint}'. "
            "Use http://, https://, dns:///, or bare host:port."
        )

    # Bare host:port (no scheme at all, or urlparse mis-detected one).
    return f"{scheme}://{endpoint}"


def _bootstrap_ssl_from_server(endpoint: str) -> bytes:
    """Fetch the server's TLS certificate chain and return it as PEM bytes.

    Used when insecure_skip_verify is enabled — trusts whatever cert
    the server presents (e.g. self-signed or corporate CA certs).

    Uses pyOpenSSL to connect with verification disabled and retrieve the
    full peer certificate chain (leaf + intermediates + root).  This works
    on all supported Python versions (>= 3.10).

    This is a blocking call.  Callers should run it via asyncio.to_thread().
    """
    hostname = hostname_from_url(endpoint)
    parts = hostname.rsplit(":", 1)
    if len(parts) == 2 and parts[1].isdigit():
        server_address = (parts[0], int(parts[1]))
    else:
        logger.warning(f"Unrecognized port in endpoint [{hostname}], defaulting to 443.")
        server_address = (hostname, 443)

    logger.debug(f"Retrieving SSL certificate chain from {server_address}")

    ctx = SSL.Context(SSL.TLS_CLIENT_METHOD)
    ctx.set_verify(SSL.VERIFY_NONE, lambda *args: True)

    sock = socket.create_connection(server_address, timeout=10)
    # create_connection with a timeout sets O_NONBLOCK on the fd. pyOpenSSL's
    # do_handshake() operates directly on the fd and raises WantReadError /
    # WantWriteError when it sees EAGAIN. settimeout(None) restores blocking
    # mode to avoid this. A positive timeout (e.g. settimeout(30)) won't help
    # because CPython still sets O_NONBLOCK for any timeout > 0.
    #
    # This means the TLS handshake has no deadline, but that is acceptable:
    # the 10s TCP connect timeout above already proves the peer is reachable,
    # this is a one-time bootstrap path (not a hot path), and it runs inside
    # asyncio.to_thread() so a stall won't block the event loop.
    sock.settimeout(None)
    conn = None
    try:
        conn = SSL.Connection(ctx, sock)
        conn.set_tlsext_host_name(server_address[0].encode())
        conn.set_connect_state()
        conn.do_handshake()

        chain = conn.get_peer_cert_chain()
        if not chain:
            raise RuntimeError(f"Server at {server_address} returned no certificates")

        pem_certs = [crypto.dump_certificate(crypto.FILETYPE_PEM, cert) for cert in chain]
        logger.debug(f"Retrieved certificate chain ({len(pem_certs)} certs) from {server_address}")
        return b"\n".join(pem_certs)
    finally:
        if conn is not None:
            conn.close()
        else:
            sock.close()


async def _resolve_tls_ca_cert(
    endpoint: str,
    *,
    insecure: bool,
    insecure_skip_verify: bool,
    ca_cert_file_path: str | None,
) -> bytes | None:
    """Determine TLS CA certificate bytes for the pyqwest transport.

    Returns PEM-encoded bytes, or None to use system defaults.
    """
    if insecure:
        return None

    if insecure_skip_verify:
        return await asyncio.to_thread(_bootstrap_ssl_from_server, endpoint)

    if ca_cert_file_path:
        import aiofiles

        async with aiofiles.open(ca_cert_file_path, "rb") as f:
            return await f.read()

    return None


def _build_pyqwest_client(tls_ca_cert: bytes | None = None) -> pyqwest.Client:
    """Build a pyqwest Client with sensible transport defaults."""
    transport = pyqwest.HTTPTransport(
        tls_ca_cert=tls_ca_cert,
        timeout=None,
        connect_timeout=30.0,
        read_timeout=None,
        pool_idle_timeout=90.0,
        tcp_keepalive_interval=30.0,  # was grpc.keepalive_time_ms = 30000
    )
    return pyqwest.Client(transport=transport)


async def create_session_config(
    endpoint: str | None,
    api_key: str | None = None,
    /,
    insecure: typing.Optional[bool] = None,
    insecure_skip_verify: typing.Optional[bool] = False,
    ca_cert_file_path: typing.Optional[str] = None,
    proxy_command: typing.List[str] | None = None,
    rpc_retries: typing.Optional[int] = None,
    auth_endpoint: typing.Optional[str] = None,
    **kwargs,
) -> SessionConfig:
    """
    Creates a SessionConfig with endpoint, interceptors, and HTTP client for ConnectRPC.

    This returns a SessionConfig namedtuple that can be used to construct
    ConnectRPC service clients.

    :param endpoint: The endpoint URL for the service
    :param api_key: API key for authentication; if provided, it will be used to detect the endpoint and credentials.
    :param insecure: Whether to use plain HTTP (no TLS)
    :param insecure_skip_verify: Whether to skip SSL certificate verification
    :param ca_cert_file_path: Path to CA certificate file for SSL verification
    :param proxy_command: List of strings for proxy command configuration
    :param rpc_retries: Number of times to retry RPCs. None means do not install the retry interceptor.
    :param auth_endpoint: Endpoint for auth/OAuth discovery. Defaults to ``endpoint`` when not set.
        When creating sessions for per-cluster DataProxy clients, pass the
        control-plane endpoint so auth tokens are obtained from the right server.
    :param kwargs: Additional arguments passed to authenticator factories
    :return: SessionConfig with endpoint, interceptors, and http_client
    """
    assert endpoint or api_key, "Either endpoint or api_key must be specified"

    if api_key:
        from flyte.remote._client.auth._auth_utils import decode_api_key

        endpoint, client_id, client_secret, _org = decode_api_key(api_key)
        kwargs["auth_type"] = "ClientSecret"
        kwargs["client_id"] = client_id
        kwargs["client_secret"] = client_secret
        kwargs["client_credentials_secret"] = client_secret

    assert endpoint, "Endpoint must be specified by this point"

    # Normalize to HTTP(S) URL
    endpoint = normalize_rpc_endpoint(endpoint, insecure=insecure or False)

    # Resolve TLS certificate for pyqwest transport (D5/D6)
    tls_ca_cert = await _resolve_tls_ca_cert(
        endpoint,
        insecure=insecure or False,
        insecure_skip_verify=insecure_skip_verify or False,
        ca_cert_file_path=ca_cert_file_path,
    )
    http_client = _build_pyqwest_client(tls_ca_cert)

    # Build interceptors list
    from ._interceptors.default_metadata import DefaultMetadataInterceptor

    interceptors: list = [DefaultMetadataInterceptor()]

    # Create httpx session for auth flows (OAuth token exchange, PKCE, etc.)
    # This is separate from the pyqwest client used for ConnectRPC transport.
    if proxy_command:
        proxy_authenticator = get_async_proxy_authenticator(endpoint=endpoint, proxy_command=proxy_command, **kwargs)
        auth_http_session = get_async_session(
            ca_cert_file_path=ca_cert_file_path, proxy_authenticator=proxy_authenticator, **kwargs
        )
        interceptors.extend(
            create_proxy_auth_interceptors(
                endpoint, proxy_command=proxy_command, http_session=auth_http_session, **kwargs
            )
        )
    else:
        auth_http_session = get_async_session(ca_cert_file_path=ca_cert_file_path, **kwargs)

    # Add auth interceptors — skip when insecure=True.
    # NOTE: insecure means "no TLS" (plain HTTP), not "no auth". However, in
    # practice a plaintext endpoint implies no auth server is available (e.g.
    # local dev). This matches the old gRPC create_channel() behavior.
    if not insecure:
        auth_interceptors = create_auth_interceptors(
            endpoint=auth_endpoint or endpoint,
            http_client=http_client,
            insecure=insecure,
            insecure_skip_verify=insecure_skip_verify,
            ca_cert_file_path=ca_cert_file_path,
            http_session=auth_http_session,
            **kwargs,
        )
        interceptors.extend(auth_interceptors)

    # Add retry interceptors
    if rpc_retries is not None and rpc_retries > 0:
        from ._interceptors.retry import RetryServerStreamInterceptor, RetryUnaryInterceptor

        interceptors.append(RetryUnaryInterceptor(max_attempts=rpc_retries + 1))
        interceptors.append(RetryServerStreamInterceptor(max_attempts=rpc_retries + 1))

    return SessionConfig(
        endpoint=endpoint,
        insecure=insecure or False,
        insecure_skip_verify=insecure_skip_verify or False,
        interceptors=tuple(interceptors),
        http_client=http_client,
        api_key=api_key,
    )
