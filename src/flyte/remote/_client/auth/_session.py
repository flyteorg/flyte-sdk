import typing
from collections import namedtuple
from urllib.parse import urlparse

import httpx

from flyte._logging import logger

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


async def create_session(
    endpoint: str | None,
    api_key: str | None = None,
    /,
    insecure: typing.Optional[bool] = None,
    insecure_skip_verify: typing.Optional[bool] = False,
    ca_cert_file_path: typing.Optional[str] = None,
    http_session: httpx.AsyncClient | None = None,
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
    :param http_session: Pre-configured HTTP session to use for requests
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

    # Build interceptors list
    from ._interceptors.default_metadata import DefaultMetadataInterceptor

    interceptors: list = [DefaultMetadataInterceptor()]

    # Create HTTP session if not provided
    if not http_session:
        proxy_authenticator = None
        if proxy_command:
            proxy_authenticator = get_async_proxy_authenticator(
                endpoint=endpoint, proxy_command=proxy_command, **kwargs
            )
        http_session = get_async_session(
            ca_cert_file_path=ca_cert_file_path, proxy_authenticator=proxy_authenticator, **kwargs
        )

    # Add proxy auth interceptors
    proxy_auth_interceptors = create_proxy_auth_interceptors(
        endpoint, proxy_command=proxy_command, http_session=http_session, **kwargs
    )
    interceptors.extend(proxy_auth_interceptors)

    # Add auth interceptors — skip when insecure=True,
    # since a plaintext connection typically means no auth server is available.
    if not insecure:
        auth_interceptors = create_auth_interceptors(
            endpoint=endpoint,
            http_client=http_session,
            insecure=insecure,
            insecure_skip_verify=insecure_skip_verify,
            ca_cert_file_path=ca_cert_file_path,
            http_session=http_session,
            **kwargs,
        )
        interceptors.extend(auth_interceptors)

    # Add retry interceptors
    if rpc_retries is not None and rpc_retries > 0:
        from ._interceptors.retry import RetryUnaryInterceptor, RetryServerStreamInterceptor

        interceptors.append(RetryUnaryInterceptor(max_attempts=rpc_retries + 1))
        interceptors.append(RetryServerStreamInterceptor(max_attempts=rpc_retries + 1))

    return SessionConfig(endpoint=endpoint, interceptors=interceptors, http_client=http_session)
