import asyncio
import dataclasses
import ssl
import typing
from abc import abstractmethod
from http import HTTPStatus

import httpx

from flyte.remote._client.auth._client_config import ClientConfig, ClientConfigStore
from flyte.remote._client.auth._keyring import Credentials, KeyringStore


@dataclasses.dataclass
class AuthHeaders:
    creds_id: str
    headers: dict[str, str]


def is_usable_access_token(access_token: typing.Optional[str]) -> bool:
    """
    Return True if ``access_token`` can be sent verbatim in an ``Authorization`` header.

    A bearer token is a single opaque string. Anything else — an empty value, an
    interactive prompt captured from a helper command's stdout, a multi-line blob, a
    stray control character — makes the header invalid, and the HTTP transport rejects
    it with an opaque ``ValueError: Invalid header value`` far from the real cause.

    :param access_token: The candidate access token
    :return: True if the token is usable as a header value
    """
    if not access_token:
        return False
    try:
        access_token.encode("latin-1")
    except UnicodeEncodeError:
        return False
    return not any(c.isspace() or ord(c) < 0x20 or ord(c) == 0x7F for c in access_token)


class Authenticator(object):
    """
    Base authenticator for all authentication flows
    """

    def __init__(
        self,
        endpoint: str,
        *,
        cfg_store: typing.Optional[ClientConfigStore] = None,
        client_config: typing.Optional[ClientConfig] = None,
        credentials: typing.Optional[Credentials] = None,
        http_session: typing.Optional[httpx.AsyncClient] = None,
        http_proxy_url: typing.Optional[str] = None,
        verify: bool = True,
        ca_cert_path: typing.Optional[str] = None,
        default_header_key: str = "authorization",
        disable_keyring: bool = False,
        **kwargs,
    ):
        """
        Initialize the base authenticator.

        Args:
            endpoint: The endpoint URL for authentication
            cfg_store: Optional client configuration store for retrieving remote configuration
            client_config: Optional client configuration containing authentication settings
            credentials: Optional credentials to use for authentication
            http_session: Optional HTTP session to use for requests
            http_proxy_url: Optional HTTP proxy URL
            verify: Whether to verify SSL certificates
            ca_cert_path: Optional path to CA certificate file
            kwargs: Additional keyword arguments passed to get_async_session, which may include:
                - auth: Authentication implementation to use
                - params: Query parameters to include in request URLs
                - headers: HTTP headers to include in requests
                - cookies: Cookies to include in requests
                - cert: SSL client certificate (path or tuple)
                - http1: Whether to enable HTTP/1.1 support
                - http2: Whether to enable HTTP/2 support
                - proxies: Proxy configuration mapping
                - mounts: Mounted transports for specific URL patterns
                - timeout: Request timeout configuration
                - follow_redirects: Whether to follow redirects
                - limits: Connection pool limits
                - max_redirects: Maximum number of redirects to follow
                - event_hooks: Event hooks for request/response lifecycle
                - base_url: Base URL to join with relative URLs
                - transport: Transport implementation to use
                - app: ASGI application to handle requests
        """
        self._endpoint = endpoint
        self._disable_keyring = disable_keyring
        self._creds = credentials or KeyringStore.retrieve(endpoint, disable=disable_keyring)
        self._http_proxy_url = http_proxy_url
        self._verify = verify
        self._ca_cert_path = ca_cert_path
        self._client_config = client_config
        self._cfg_store = cfg_store
        # Will be populated by _ensure_remote_config
        self._resolved_config: ClientConfig | None = None
        # Lock for coroutine safety
        self._async_lock = asyncio.Lock()
        self._http_session = http_session or get_async_session(**kwargs)
        # Id for tracking credential refresh state
        self._creds_id = self._creds.id if self._creds else None
        self._default_header_key = default_header_key

    async def _resolve_config(self) -> ClientConfig:
        """
        Resolves and merges client configuration with remote configuration.

        This method fetches the remote configuration from the cfg_store and merges it with
        the local client_config, prioritizing local settings over remote ones.

        This method is thread-safe and coroutine-safe, ensuring the remote config is fetched
        only once regardless of concurrent access from multiple threads or coroutines.

        Returns:
            A merged ClientConfig object containing resolved configuration settings
        """
        # First check without locks for performance
        if self._resolved_config is not None:
            return self._resolved_config

        if self._cfg_store is None:
            raise ValueError("ClientConfigStore is not set. Cannot resolve configuration.")

        remote_config = await self._cfg_store.get_client_config()
        self._resolved_config = (
            remote_config.with_override(self._client_config) if self._client_config else remote_config
        )

        return self._resolved_config

    def get_credentials(self) -> typing.Optional[Credentials]:
        """
        Get the current credentials.

        Returns:
            The current credentials or None if not set
        """
        return self._creds

    def _set_credentials(self, creds: Credentials):
        """
        Set the credentials.

        Args:
            creds: The credentials to set
        """
        self._creds = creds

    async def get_auth_headers(self) -> typing.Optional[AuthHeaders]:
        """
        Fetch the authentication headers.

        Returns:
            AuthHeaders with header dict, or None if no credentials are available
        """
        creds = self.get_credentials()
        if creds:
            if not is_usable_access_token(creds.access_token):
                # A malformed token can reach us from the keyring, where it was cached by
                # an earlier run before it was ever used in a request. Treating it as "no
                # credentials" lets the normal unauthenticated path trigger a refresh,
                # which re-runs the auth flow and surfaces a real error, instead of
                # wedging the user until they clear the keyring by hand.
                from flyte._logging import logger

                logger.warning(
                    f"Ignoring cached credentials for {self._endpoint}: the stored access token cannot be sent"
                    f" as an Authorization header. Re-authenticating."
                )
                return None

            header_key = self._default_header_key
            if self._resolved_config is not None:
                # We only resolve the config during authentication flow, to avoid unnecessary network calls
                # and usually the header_key is consistent.
                header_key = self._resolved_config.header_key
            return AuthHeaders(
                creds_id=creds.id,
                headers={header_key: f"Bearer {creds.access_token}"},
            )
        return None

    async def refresh_credentials(self, creds_id: str | None = None):
        """
        Refresh the credentials asynchronously with thread and asyncio safety.

        This method implements a thread-safe and coroutine-safe credential refresh mechanism.
        It uses a timestamp-based approach to prevent redundant credential refreshes when
        multiple threads or coroutines attempt to refresh credentials simultaneously.

        The caller should capture the current _creds_timestamp before attempting to use credentials.
        If credential usage fails, the caller can pass that timestamp to this method.
        If the timestamp matches the current value, a refresh is needed; otherwise,
        another thread has already refreshed the credentials.

        Args:
            creds_id: The id of credentials when they were last accessed by the caller.
                If None, force a refresh regardless of id.

        Raises:
            Exception: Authentication-related exceptions if the refresh fails.
        """
        # If creds_id is None, force refresh
        # If creds_id matches current value, credentials need refresh
        # If creds_id doesn't match, another thread already refreshed credentials
        if creds_id and creds_id != self._creds_id:
            # Credentials have been refreshed by another thread/coroutine since caller read them
            return

        # Use the async lock to ensure coroutine safety
        async with self._async_lock:
            # Double-check pattern to avoid unnecessary work
            if creds_id and creds_id != self._creds_id:
                # Another thread/coroutine refreshed credentials while we were waiting for the lock
                return

            # Perform the actual credential refresh
            try:
                self._creds = await self._do_refresh_credentials()
                KeyringStore.store(self._creds, disable=self._disable_keyring)
            except Exception:
                KeyringStore.delete(self._endpoint, disable=self._disable_keyring)
                raise

            # Update the timestamp to indicate credentials have been refreshed
            self._creds_id = self._creds.id

    @abstractmethod
    async def _do_refresh_credentials(self) -> Credentials:
        """
        Perform the actual credential refresh operation.

        This method must be implemented by subclasses to handle the specific authentication flow.
        It should update the internal credentials object (_creds) with a new access token.

        Implementations typically use the resolved configuration from _resolve_config() to
        determine authentication endpoints, scopes, audience, and other parameters needed for
        the specific authentication flow.

        Raises:
            Exception: Authentication-related exceptions if the refresh fails.
        """
        ...


class AsyncAuthenticatedClient(httpx.AsyncClient):
    """
    An httpx.AsyncClient that automatically adds authentication headers to requests.
    This class extends httpx.AsyncClient which is inherently async for network operations.
    """

    def __init__(self, authenticator: Authenticator, **kwargs):
        """
        Initialize the authenticated client.

        Args:
            authenticator: The authenticator to use for authentication
            kwargs: Additional arguments passed to the httpx.AsyncClient constructor
        """
        super().__init__(**kwargs)
        self.auth_adapter = AsyncAuthenticationHTTPAdapter(authenticator)
        self.authenticator = authenticator

    async def send(self, request: httpx.Request, **kwargs) -> httpx.Response:
        """
        Sends the request with added authentication headers.
        Must be async because it performs network IO operations and may need to refresh credentials.
        If the response returns a 401 status code, refreshes the credentials and retries the request.

        Args:
            request: The request object to send.
            kwargs: Additional keyword arguments passed to the parent httpx.AsyncClient.send method, which may
                include:
                - auth: Authentication implementation to use for this request
                - follow_redirects: Whether to follow redirects for this request
                - timeout: Request timeout configuration for this request

        Returns:
            The response object.
        """

        creds_id = await self.auth_adapter.add_auth_header(request)
        response = await super().send(request, **kwargs)

        if response.status_code == HTTPStatus.UNAUTHORIZED:
            await self.authenticator.refresh_credentials(creds_id=creds_id)
            await self.auth_adapter.add_auth_header(request)
            response = await super().send(request, **kwargs)

        return response


class AsyncAuthenticationHTTPAdapter:
    """
    A custom async HTTP adapter that adds authentication headers to requests of an httpx.AsyncClient.
    This is the async equivalent of AuthenticationHTTPAdapter for requests.
    """

    def __init__(self, authenticator: Authenticator):
        """
        Initialize the authentication HTTP adapter.

        Args:
            authenticator: The authenticator to use for authentication
        """
        self.authenticator = authenticator

    async def add_auth_header(self, request: httpx.Request) -> typing.Optional[str]:
        """
        Adds authentication headers to the request.
        Must be async because it may call refresh_credentials which performs IO operations.

        Args:
            request: The request object to add headers to.

        Returns:
            The credentials ID (creds_id) used for tracking credential refresh state
        """
        if self.authenticator.get_credentials() is None:
            await self.authenticator.refresh_credentials()

        metadata = await self.authenticator.get_auth_headers()
        if metadata is None:
            return None
        for key, value in metadata.headers.items():
            request.headers[key] = value
        return metadata.creds_id


def upgrade_async_session_to_proxy_authenticated(
    http_session: httpx.AsyncClient, proxy_authenticator: typing.Optional[Authenticator] = None, **kwargs
) -> httpx.AsyncClient:
    """
    Given an httpx.AsyncClient, it returns a new session that uses AsyncAuthenticationHTTPAdapter
    to perform authentication with a proxy in front of Flyte

    Args:
        http_session: httpx.AsyncClient Precreated session
        proxy_authenticator: Optional authenticator for proxy authentication
        kwargs: Additional arguments passed to AsyncAuthenticatedClient, which may include:
            - auth: Authentication implementation to use
            - params: Query parameters to include in request URLs
            - headers: HTTP headers to include in requests
            - cookies: Cookies to include in requests
            - verify: SSL verification mode (True/False/path to certificate)
            - cert: SSL client certificate (path or tuple)
            - http1: Whether to enable HTTP/1.1 support
            - http2: Whether to enable HTTP/2 support
            - proxies: Proxy configuration mapping
            - mounts: Mounted transports for specific URL patterns
            - timeout: Request timeout configuration
            - follow_redirects: Whether to follow redirects
            - limits: Connection pool limits
            - max_redirects: Maximum number of redirects to follow
            - event_hooks: Event hooks for request/response lifecycle
            - base_url: Base URL to join with relative URLs
            - transport: Transport implementation to use
            - app: ASGI application to handle requests

    Returns:
        httpx.AsyncClient with authentication
    """
    if proxy_authenticator:
        return AsyncAuthenticatedClient(proxy_authenticator, **kwargs)
    else:
        return http_session


def get_async_session(
    proxy_authenticator: Authenticator | None = None,
    ca_cert_path: str | None = None,
    verify: bool | None = None,
    **kwargs,
) -> httpx.AsyncClient:
    """
    Returns a new httpx.AsyncClient with proxy authentication if proxy_authenticator is provided.

    This function creates a new httpx.AsyncClient and optionally configures it with proxy authentication
    if a proxy authenticator is provided.

    Args:
        proxy_authenticator: Optional authenticator for proxy authentication
        ca_cert_path: Optional path to CA certificate file for SSL verification
        verify: Optional SSL verification mode (True/False/path to certificate)
        kwargs: Additional keyword arguments passed to httpx.AsyncClient constructor and AsyncAuthenticatedClient,
            which may include:
               - auth: Authentication implementation to use
               - params: Query parameters to include in request URLs
               - headers: HTTP headers to include in requests
               - cookies: Cookies to include in requests
               - cert: SSL client certificate (path or tuple)
               - http1: Whether to enable HTTP/1.1 support
               - http2: Whether to enable HTTP/2 support
               - proxies: Proxy configuration mapping
               - mounts: Mounted transports for specific URL patterns
               - timeout: Request timeout configuration
               - follow_redirects: Whether to follow redirects
               - limits: Connection pool limits
               - max_redirects: Maximum number of redirects to follow
               - event_hooks: Event hooks for request/response lifecycle
               - base_url: Base URL to join with relative URLs
               - transport: Transport implementation to use
               - app: ASGI application to handle requests
               - proxy_env: Environment variables for proxy command
               - proxy_timeout: Timeout for proxy command execution
               - header_key: Header key to use for authentication
               - endpoint: The endpoint URL for authentication
               - client_id: Client ID for authentication
               - client_secret: Client secret for authentication
               - scopes: List of scopes to request during authentication
               - audience: Audience for the token
               - http_proxy_url: HTTP proxy URL

    Returns:
        An httpx.AsyncClient instance, optionally configured with proxy authentication
    """

    # Extract known httpx.AsyncClient parameters from kwargs
    client_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        in [
            "auth",
            "params",
            "headers",
            "cookies",
            "verify",
            "cert",
            "http1",
            "http2",
            "proxies",
            "mounts",
            "timeout",
            "follow_redirects",
            "limits",
            "max_redirects",
            "event_hooks",
            "base_url",
            "transport",
            "app",
        ]
    }

    if ca_cert_path:
        context = ssl.create_default_context(capath=ca_cert_path)
        verify = True if context is not None else False

    if verify is not None:
        client_kwargs["verify"] = verify

    http_session = httpx.AsyncClient(**client_kwargs)
    if proxy_authenticator:
        http_session = upgrade_async_session_to_proxy_authenticated(
            http_session, proxy_authenticator=proxy_authenticator, **kwargs
        )
    return http_session
