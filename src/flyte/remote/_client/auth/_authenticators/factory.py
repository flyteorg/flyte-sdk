import typing

import grpc.aio

from flyte.remote._client.auth._authenticators.base import Authenticator
from flyte.remote._client.auth._authenticators.external_command import (
    AsyncCommandAuthenticator,
)
from flyte.remote._client.auth._client_config import AuthType, ClientConfigStore, RemoteClientConfigStore


def create_auth_interceptors(
    endpoint: str, in_channel: grpc.aio.Channel, **kwargs
) -> typing.List[grpc.aio.ClientInterceptor]:
    """
    Async version of upgrade_channel_to_authenticated.
    Given a grpc.Channel, preferably a secure channel, it returns a list of interceptors to
    perform an Oauth2.0 Auth flow for all RPC call types.

    :param endpoint: The endpoint URL for authentication
    :param in_channel: grpc.Channel Precreated channel
    :param kwargs: Additional arguments passed to the authenticator, including:
        - insecure: Whether to use an insecure channel
        - insecure_skip_verify: Whether to skip SSL certificate verification
        - ca_cert_file_path: Path to CA certificate file for SSL verification
        - auth_type: The authentication type to use ("Pkce", "ClientSecret", "ExternalCommand", "DeviceFlow")
        - command: Command to execute for ExternalCommand authentication
        - client_id: Client ID for ClientSecret authentication
        - client_secret: Client secret for ClientSecret authentication
        - scopes: List of scopes to request during authentication
        - audience: Audience for the token
        - http_proxy_url: HTTP proxy URL
        - http_session: httpx.AsyncClient session
        - verify: Whether to verify SSL certificates
        - ca_cert_path: Optional path to CA certificate file
        - header_key: Header key to use for authentication
        - proxy_env: Environment variables for proxy command
        - proxy_timeout: Timeout for proxy command execution
        - redirect_uri: OAuth2 redirect URI for PKCE authentication
        - add_request_auth_code_params_to_request_access_token_params: Whether to add auth code params to token request
        - request_auth_code_params: Parameters to add to login URI opened in browser
        - request_access_token_params: Parameters to add when exchanging auth code for access token
        - refresh_access_token_params: Parameters to add when refreshing access token
    :return: List of gRPC interceptors for different call types
    """
    from flyte.remote._client.auth._grpc_utils.auth_interceptor import (
        AuthStreamStreamInterceptor,
        AuthStreamUnaryInterceptor,
        AuthUnaryStreamInterceptor,
        AuthUnaryUnaryInterceptor,
    )

    def authenticator_factory() -> Authenticator:
        return get_async_authenticator(endpoint=endpoint, cfg_store=RemoteClientConfigStore(in_channel), **kwargs)

    return [
        AuthUnaryUnaryInterceptor(authenticator_factory),
        AuthUnaryStreamInterceptor(authenticator_factory),
        AuthStreamUnaryInterceptor(authenticator_factory),
        AuthStreamStreamInterceptor(authenticator_factory),
    ]


def create_proxy_auth_interceptors(
    endpoint: str, proxy_command: typing.Optional[typing.List[str]] = None, **kwargs
) -> typing.List[grpc.aio.ClientInterceptor]:
    """
    Async version of upgrade_channel_to_proxy_authenticated.
    If activated in the platform config, given a grpc.Channel, preferably a secure channel, it returns a list of
    interceptors to perform authentication with a proxy in front of Flyte for all RPC call types.

    :param endpoint: The endpoint URL for authentication
    :param proxy_command: Command to execute to get proxy authentication token
    :param kwargs: Additional arguments passed to the authenticator, including:
        - proxy_env: Environment variables for the proxy command
        - proxy_timeout: Timeout for the proxy command
        - header_key: Header key to use for authentication (defaults to "proxy-authorization")
        - http_session: httpx.AsyncClient session to use for requests
        - verify: Whether to verify SSL certificates
        - ca_cert_path: Optional path to CA certificate file
    :return: List of gRPC interceptors for different call types
    """
    if proxy_command:
        from flyte.remote._client.auth._grpc_utils.auth_interceptor import (
            AuthStreamStreamInterceptor,
            AuthStreamUnaryInterceptor,
            AuthUnaryStreamInterceptor,
            AuthUnaryUnaryInterceptor,
        )

        def authenticator_factory() -> Authenticator:
            return get_async_proxy_authenticator(endpoint=endpoint, proxy_command=proxy_command, **kwargs)

        return [
            AuthUnaryUnaryInterceptor(authenticator_factory),
            AuthUnaryStreamInterceptor(authenticator_factory),
            AuthStreamUnaryInterceptor(authenticator_factory),
            AuthStreamStreamInterceptor(authenticator_factory),
        ]
    else:
        return []


def get_async_authenticator(
    endpoint: str,
    cfg_store: ClientConfigStore,
    *,
    command: typing.Optional[typing.List[str]] = None,
    insecure_skip_verify: bool = False,
    auth_type: AuthType = "Pkce",
    ca_cert_file_path: typing.Optional[str] = None,
    **kwargs,
) -> Authenticator:
    """
    Returns a new authenticator based on the platform config.
    This is an async-compatible version of get_authenticator.
    Must be async because it calls get_async_session which may perform IO operations.

    :param endpoint: The endpoint URL for authentication
    :param cfg_store: The client configuration store
    :param command: Command to execute for ExternalCommand authentication
    :param insecure_skip_verify: Whether to skip SSL certificate verification
    :param auth_type: The authentication type to use
    :param ca_cert_file_path: Path to CA certificate file for SSL verification
    :param kwargs: Additional arguments passed to the authenticator, which may include:
        - http_session: httpx.AsyncClient session to use for requests
        - client_config: Optional client configuration containing authentication settings
        - credentials: Optional credentials to use for authentication
        - http_proxy_url: HTTP proxy URL
        - verify: Whether to verify SSL certificates (bool or path to cert)
        - ca_cert_path: Optional path to CA certificate file
        - client_id: Client ID for ClientSecret authentication
        - client_secret: Client secret for ClientSecret authentication (for ClientSecret auth)
        - client_credentials_secret: Client secret for ClientSecret authentication (alias)
        - scopes: List of scopes to request during authentication
        - audience: Audience for the token
        - header_key: Header key to use for authentication
        - proxy_env: Environment variables for proxy command
        - proxy_timeout: Timeout for proxy command execution
        - redirect_uri: OAuth2 redirect URI for PKCE authentication
        - add_request_auth_code_params_to_request_access_token_params: Whether to add auth code params to token request
        - request_auth_code_params: Parameters to add to login URI opened in browser
        - request_access_token_params: Parameters to add when exchanging auth code for access token
        - refresh_access_token_params: Parameters to add when refreshing access token
    :return: An authenticator instance
    """
    verify = None
    if insecure_skip_verify:
        verify = False
    elif ca_cert_file_path:
        verify = True if ca_cert_file_path is not None else False

    # Note: The authenticator classes already have async refresh_credentials methods
    # so we can reuse them with our async session
    match auth_type:
        case "Pkce":
            from flyte.remote._client.auth._authenticators.pkce import PKCEAuthenticator

            return PKCEAuthenticator(endpoint=endpoint, cfg_store=cfg_store, verify=verify, **kwargs)
        case "ClientSecret":
            from flyte.remote._client.auth._authenticators.client_credentials import ClientCredentialsAuthenticator

            return ClientCredentialsAuthenticator(endpoint=endpoint, cfg_store=cfg_store, verify=verify, **kwargs)
        case "ExternalCommand":
            from flyte.remote._client.auth._authenticators.external_command import AsyncCommandAuthenticator

            return AsyncCommandAuthenticator(endpoint=endpoint, command=command, verify=verify, **kwargs)
        case "DeviceFlow":
            from flyte.remote._client.auth._authenticators.device_code import DeviceCodeAuthenticator

            return DeviceCodeAuthenticator(endpoint=endpoint, cfg_store=cfg_store, verify=verify, **kwargs)
        case _:
            raise ValueError(
                f"Invalid auth mode [{auth_type}] specified. Please update the creds config to use a valid value"
            )


def get_async_proxy_authenticator(endpoint: str, *, proxy_command: typing.List[str], **kwargs) -> Authenticator:
    """
    Returns an async authenticator for proxy authentication.
    This function needs to be async because it calls get_async_command_authenticator which performs IO operations.

    :param endpoint: The endpoint URL for authentication
    :param proxy_command: Command to execute to get proxy authentication token
    :param kwargs: Additional arguments passed to the authenticator, including:
        - header_key: Header key to use for authentication (defaults to "proxy-authorization")
        - proxy_env: Environment variables for the proxy command
        - proxy_timeout: Timeout for the proxy command
        - http_session: httpx.AsyncClient session to use for requests
        - cfg_store: Optional client configuration store for retrieving remote configuration
        - client_config: Optional client configuration containing authentication settings
        - credentials: Optional credentials to use for authentication
        - http_proxy_url: Optional HTTP proxy URL
        - verify: Whether to verify SSL certificates (default: True)
        - ca_cert_path: Optional path to CA certificate file
    :return: An authenticator instance for proxy authentication
    """
    return AsyncCommandAuthenticator(
        endpoint=endpoint, command=proxy_command, header_key="proxy-authorization", **kwargs
    )
