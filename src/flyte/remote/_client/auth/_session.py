import asyncio
import ssl
import typing
from collections import namedtuple
from urllib.parse import urlparse

import pyqwest

from flyte._logging import logger
from flyte._utils.org_discovery import hostname_from_url

from ._authenticators.base import get_async_session
from ._authenticators.factory import create_auth_interceptors, create_proxy_auth_interceptors, get_async_proxy_authenticator

SessionConfig = namedtuple("SessionConfig", ["endpoint", "interceptors", "http_client"])


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
    """Fetch the server's TLS certificate and return it as PEM bytes.

    Used when insecure_skip_verify is enabled — trusts whatever cert
    the server presents (e.g. self-signed certs in dev/staging).

    This is a blocking call (ssl.get_server_certificate uses sockets).
    Callers should run it via asyncio.to_thread().
    """
    hostname = hostname_from_url(endpoint)
    parts = hostname.rsplit(":", 1)
    if len(parts) == 2 and parts[1].isdigit():
        server_address = (parts[0], int(parts[1]))
    else:
        logger.warning(f"Unrecognized port in endpoint [{hostname}], defaulting to 443.")
        server_address = (hostname, 443)

    logger.debug(f"Retrieving SSL certificate from {server_address}")
    cert = ssl.get_server_certificate(server_address, timeout=10)
    return cert.encode()


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
    """Build a pyqwest Client with transport defaults matching the old gRPC channel config.

    These defaults are always set explicitly so behaviour doesn't silently
    change if pyqwest changes its own defaults in a future release.

    Mapping from old gRPC channel options:
        grpc.keepalive_time_ms = 30000        → tcp_keepalive_interval = 30.0
        grpc.keepalive_timeout_ms = 10000     → (OS-level TCP; no pyqwest knob)
        grpc.keepalive_permit_without_calls=1 → pyqwest keepalive is always-on
        pool idle timeout (implicit 2 min)    → pool_idle_timeout = 90.0
        connection timeout (implicit)         → connect_timeout = 30.0
    """
    transport = pyqwest.HTTPTransport(
        tls_ca_cert=tls_ca_cert,
        tcp_keepalive_interval=30.0,
        pool_idle_timeout=90.0,
        connect_timeout=30.0,
    )
    return pyqwest.Client(transport=transport)


async def create_session(
    endpoint: str | None,
    api_key: str | None = None,
    /,
    insecure: typing.Optional[bool] = None,
    insecure_skip_verify: typing.Optional[bool] = False,
    ca_cert_file_path: typing.Optional[str] = None,
    proxy_command: typing.List[str] | None = None,
    rpc_retries: typing.Optional[int] = None,
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
    proxy_authenticator = None
    if proxy_command:
        proxy_authenticator = get_async_proxy_authenticator(
            endpoint=endpoint, proxy_command=proxy_command, **kwargs
        )
    auth_http_session = get_async_session(
        ca_cert_file_path=ca_cert_file_path, proxy_authenticator=proxy_authenticator, **kwargs
    )

    # Add proxy auth interceptors
    proxy_auth_interceptors = create_proxy_auth_interceptors(
        endpoint, proxy_command=proxy_command, http_session=auth_http_session, **kwargs
    )
    interceptors.extend(proxy_auth_interceptors)

    # Add auth interceptors — skip when insecure=True.
    # NOTE: insecure means "no TLS" (plain HTTP), not "no auth". However, in
    # practice a plaintext endpoint implies no auth server is available (e.g.
    # local dev). This matches the old gRPC create_channel() behavior.
    if not insecure:
        auth_interceptors = create_auth_interceptors(
            endpoint=endpoint,
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
        from ._interceptors.retry import RetryUnaryInterceptor, RetryServerStreamInterceptor

        interceptors.append(RetryUnaryInterceptor(max_attempts=rpc_retries + 1))
        interceptors.append(RetryServerStreamInterceptor(max_attempts=rpc_retries + 1))

    return SessionConfig(endpoint=endpoint, interceptors=interceptors, http_client=http_client)
