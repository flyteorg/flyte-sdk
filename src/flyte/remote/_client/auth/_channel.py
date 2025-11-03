import ssl
import typing

import grpc
import grpc.aio
import httpx
from grpc.experimental.aio import init_grpc_aio

from flyte._logging import logger
from flyte._utils.org_discovery import hostname_from_url

from ._authenticators.base import get_async_session
from ._authenticators.factory import (
    create_auth_interceptors,
    create_proxy_auth_interceptors,
    get_async_proxy_authenticator,
)

# Initialize gRPC AIO early enough so it can be used in the main thread
init_grpc_aio()


def bootstrap_ssl_from_server(endpoint: str) -> grpc.ChannelCredentials:
    """
    Retrieves the SSL certificate from the remote server and creates gRPC channel credentials.

    This function should be used only when insecure-skip-verify is enabled. It extracts the server address
    and port from the endpoint URL, retrieves the SSL certificate from the server, and creates
    gRPC channel credentials using the certificate.

    :param endpoint: The endpoint URL to retrieve the SSL certificate from, may include port number
    :return: gRPC channel credentials created from the retrieved certificate
    """
    hostname = hostname_from_url(endpoint)

    # Get port from endpoint or use 443
    endpoint_parts = hostname.rsplit(":", 1)
    if len(endpoint_parts) == 2 and endpoint_parts[1].isdigit():
        server_address = (endpoint_parts[0], int(endpoint_parts[1]))
    else:
        logger.warning(f"Unrecognized port in endpoint [{hostname}], defaulting to 443.")
        server_address = (hostname, 443)

    # Run the blocking SSL certificate retrieval with a timeout
    logger.debug(f"Retrieving SSL certificate from {server_address}")
    cert = ssl.get_server_certificate(server_address, timeout=10)
    return grpc.ssl_channel_credentials(str.encode(cert))


async def create_channel(
    endpoint: str | None,
    api_key: str | None = None,
    /,
    insecure: typing.Optional[bool] = None,
    insecure_skip_verify: typing.Optional[bool] = False,
    ca_cert_file_path: typing.Optional[str] = None,
    ssl_credentials: typing.Optional[grpc.ssl_channel_credentials] = None,
    grpc_options: typing.Optional[typing.Sequence[typing.Tuple[str, typing.Any]]] = None,
    compression: typing.Optional[grpc.Compression] = None,
    http_session: httpx.AsyncClient | None = None,
    proxy_command: typing.List[str] | None = None,
    **kwargs,
) -> grpc.aio.Channel:
    """
    Creates a new gRPC channel with appropriate authentication interceptors.

    This function creates either a secure or insecure gRPC channel based on the provided parameters,
    and adds authentication interceptors to the channel. If SSL credentials are not provided,
    they are created based on the insecure_skip_verify and ca_cert_file_path parameters.

    The function is async because it may need to read certificate files asynchronously
    and create authentication interceptors that perform async operations.

    :param endpoint: The endpoint URL for the gRPC channel
    :param api_key: API key for authentication; if provided, it will be used to detect the endpoint and credentials.
    :param insecure: Whether to use an insecure channel (no SSL)
    :param insecure_skip_verify: Whether to skip SSL certificate verification
    :param ca_cert_file_path: Path to CA certificate file for SSL verification
    :param ssl_credentials: Pre-configured SSL credentials for the channel
    :param grpc_options: Additional gRPC channel options
    :param compression: Compression method for the channel
    :param http_session: Pre-configured HTTP session to use for requests
    :param proxy_command: List of strings for proxy command configuration
    :param kwargs: Additional arguments passed to various functions:
        - For grpc.aio.insecure_channel/secure_channel:
            - root_certificates: Root certificates for SSL credentials
            - private_key: Private key for SSL credentials
            - certificate_chain: Certificate chain for SSL credentials
            - options: gRPC channel options
            - compression: gRPC compression method
        - For proxy configuration:
            - proxy_env: Dict of environment variables for proxy
            - proxy_timeout: Timeout for proxy connection
        - For authentication interceptors (passed to create_auth_interceptors and create_proxy_auth_interceptors):
            - auth_type: The authentication type to use ("Pkce", "ClientSecret", "ExternalCommand", "DeviceFlow")
            - command: Command to execute for ExternalCommand authentication
            - client_id: Client ID for ClientSecret authentication
            - client_secret: Client secret for ClientSecret authentication
            - client_credentials_secret: Client secret for ClientSecret authentication (alias)
            - scopes: List of scopes to request during authentication
            - audience: Audience for the token
            - http_proxy_url: HTTP proxy URL
            - verify: Whether to verify SSL certificates
            - ca_cert_path: Optional path to CA certificate file
            - header_key: Header key to use for authentication
            - redirect_uri: OAuth2 redirect URI for PKCE authentication
            - add_request_auth_code_params_to_request_access_token_params: Whether to add auth code params to token
                request
            - request_auth_code_params: Parameters to add to login URI opened in browser
            - request_access_token_params: Parameters to add when exchanging auth code for access token
            - refresh_access_token_params: Parameters to add when refreshing access token
    :return: grpc.aio.Channel with authentication interceptors configured
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

    if not ssl_credentials:
        if insecure_skip_verify:
            ssl_credentials = bootstrap_ssl_from_server(endpoint)
        elif ca_cert_file_path:
            import aiofiles

            async with aiofiles.open(ca_cert_file_path, "rb") as f:
                st_cert = await f.read()
            ssl_credentials = grpc.ssl_channel_credentials(st_cert)
        else:
            ssl_credentials = grpc.ssl_channel_credentials()

    # Create an unauthenticated channel first to use to get the server metadata
    if insecure:
        insecure_kwargs = {}
        if kw_opts := kwargs.get("options"):
            insecure_kwargs["options"] = kw_opts
        if compression:
            insecure_kwargs["compression"] = compression
        unauthenticated_channel = grpc.aio.insecure_channel(endpoint, **insecure_kwargs)
    else:
        unauthenticated_channel = grpc.aio.secure_channel(
            target=endpoint,
            credentials=ssl_credentials,
            options=grpc_options,
            compression=compression,
        )

    from ._grpc_utils.default_metadata_interceptor import (
        DefaultMetadataStreamStreamInterceptor,
        DefaultMetadataStreamUnaryInterceptor,
        DefaultMetadataUnaryStreamInterceptor,
        DefaultMetadataUnaryUnaryInterceptor,
    )

    # Add all types of default metadata interceptors
    interceptors: typing.List[grpc.aio.ClientInterceptor] = [
        DefaultMetadataUnaryUnaryInterceptor(),
        DefaultMetadataUnaryStreamInterceptor(),
        DefaultMetadataStreamUnaryInterceptor(),
        DefaultMetadataStreamStreamInterceptor(),
    ]

    # Create an HTTP session if not provided so we share the same http client across the stack
    if not http_session:
        proxy_authenticator = None
        if proxy_command:
            proxy_authenticator = get_async_proxy_authenticator(
                endpoint=endpoint, proxy_command=proxy_command, **kwargs
            )

        http_session = get_async_session(
            ca_cert_file_path=ca_cert_file_path, proxy_authenticator=proxy_authenticator, **kwargs
        )

    # Get proxy auth interceptors
    proxy_auth_interceptors = create_proxy_auth_interceptors(endpoint, http_session=http_session, **kwargs)
    interceptors.extend(proxy_auth_interceptors)

    # Get auth interceptors
    auth_interceptors = create_auth_interceptors(
        endpoint=endpoint,
        in_channel=unauthenticated_channel,
        insecure=insecure,
        insecure_skip_verify=insecure_skip_verify,
        ca_cert_file_path=ca_cert_file_path,
        http_session=http_session,
        **kwargs,
    )

    interceptors.extend(auth_interceptors)

    if insecure:
        insecure_kwargs = {}
        if kw_opts := kwargs.get("options"):
            insecure_kwargs["options"] = kw_opts
        if compression:
            insecure_kwargs["compression"] = compression
        return grpc.aio.insecure_channel(endpoint, interceptors=interceptors, **insecure_kwargs)

    return grpc.aio.secure_channel(
        target=endpoint,
        credentials=ssl_credentials,
        options=grpc_options,
        compression=compression,
        interceptors=interceptors,
    )
