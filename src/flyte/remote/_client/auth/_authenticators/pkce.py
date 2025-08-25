from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import re
import threading
import time
import typing
import webbrowser
from http import HTTPStatus as _StatusCodes
from queue import Queue
from urllib import parse as _urlparse
from urllib.parse import urlencode as _urlencode

import click
import httpx
import pydantic
from h11 import Response

from flyte._logging import logger
from flyte.remote._client.auth._authenticators.base import Authenticator
from flyte.remote._client.auth._default_html import get_default_success_html
from flyte.remote._client.auth._keyring import Credentials
from flyte.remote._client.auth.errors import AccessTokenNotFoundError

_utf_8 = "utf-8"
_code_verifier_length = 64
_random_seed_length = 40


class PKCEAuthenticator(Authenticator):
    """
    This Authenticator encapsulates the entire PKCE flow and automatically opens a browser window for login

    For Auth0 - you will need to manually configure your config.yaml to include a scopes list of the syntax:
    admin.scopes: ["offline_access", "offline", "all", "openid"] and/or similar scopes in order to get the refresh
    token + caching. Otherwise, it will just receive the access token alone. Your FlyteCTL Helm config however should
    only contain ["offline", "all"] - as OIDC scopes are not-grantable in Auth0 customer APIs. They are simply requested
    for in the POST request during the token caching process.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialize with default creds from KeyStore using the endpoint name

        :param kwargs: Keyword arguments passed to the base Authenticator

        **Keyword Arguments passed to base Authenticator**:
        :param endpoint: The endpoint URL for authentication (required)
        :param cfg_store: Optional client configuration store for retrieving remote configuration
        :param client_config: Optional client configuration containing authentication settings
        :param credentials: Optional credentials to use for authentication
        :param http_session: Optional HTTP session to use for requests
        :param http_proxy_url: Optional HTTP proxy URL
        :param verify: Whether to verify SSL certificates (default: True)
        :param ca_cert_path: Optional path to CA certificate file
        :param client_id: Client ID for authentication
        :param scopes: List of scopes to request during authentication
        :param audience: Audience for the token
        :param redirect_uri: OAuth2 redirect URI for authentication
        :param authorization_endpoint: Authorization endpoint for OAuth2 flow
        :param token_endpoint: Token endpoint for OAuth2 flow
        :param add_request_auth_code_params_to_request_access_token_params: Whether to add auth code params to token
            request
        :param request_auth_code_params: Parameters to add to login URI opened in browser
        :param request_access_token_params: Parameters to add when exchanging auth code for access token
        :param refresh_access_token_params: Parameters to add when refreshing access token
        """
        super().__init__(**kwargs)
        self._auth_client = None

    async def _initialize_auth_client(self):
        if not self._auth_client:
            code_verifier = await _generate_code_verifier()
            code_challenge = await _create_code_challenge(code_verifier)

            cfg = await self._resolve_config()
            self._auth_client = AuthorizationClient(
                endpoint=self._endpoint,
                redirect_uri=cfg.redirect_uri,
                client_id=cfg.client_id,
                # Audience only needed for Auth0 - Taken from client config
                audience=cfg.audience,
                scopes=cfg.scopes,
                # self._scopes refers to flytekit.configuration.PlatformConfig (config.yaml)
                # cfg.scopes refers to PublicClientConfig scopes (can be defined in Helm deployments)
                auth_endpoint=cfg.authorization_endpoint,
                token_endpoint=cfg.token_endpoint,
                verify=self._verify,
                http_session=self._http_session,
                request_auth_code_params={
                    "code_challenge": code_challenge,
                    "code_challenge_method": "S256",
                },
                request_access_token_params={
                    "code_verifier": code_verifier,
                },
                refresh_access_token_params={},
                add_request_auth_code_params_to_request_access_token_params=True,
            )

    async def _do_refresh_credentials(self) -> Credentials:
        """
        Refreshes the authentication credentials using PKCE flow.

        First attempts to refresh using a refresh token if available.
        If that fails or if no credentials exist, initiates the full PKCE authorization flow,
        which typically involves opening a browser for user authentication.

        This method initializes the auth client if needed, then attempts to refresh or acquire
        new credentials, and updates the internal credentials object.

        :raises: May raise authentication-related exceptions if the refresh fails
        """
        await self._initialize_auth_client()
        if self._creds:
            """We have an access token so lets try to refresh it"""
            try:
                return await self._auth_client.refresh_access_token(self._creds)
            except AccessTokenNotFoundError:
                logger.warning("Logging in...")

        return await self._auth_client.get_creds_from_remote()


class AuthorizationClient(object):
    """
    Authorization client that stores the credentials in keyring and uses oauth2 standard flow to retrieve the
    credentials. NOTE: This will open an web browser to retrieve the credentials.
    """

    def __init__(
        self,
        endpoint: str,
        auth_endpoint: str,
        token_endpoint: str,
        http_session: httpx.AsyncClient,
        audience: typing.Optional[str] = None,
        scopes: typing.Optional[typing.List[str]] = None,
        client_id: typing.Optional[str] = None,
        redirect_uri: typing.Optional[str] = None,
        endpoint_metadata: typing.Optional[EndpointMetadata] = None,
        verify: bool = True,
        ca_cert_path: typing.Optional[str] = None,
        request_auth_code_params: typing.Optional[typing.Dict[str, str]] = None,
        request_access_token_params: typing.Optional[typing.Dict[str, str]] = None,
        refresh_access_token_params: typing.Optional[typing.Dict[str, str]] = None,
        add_request_auth_code_params_to_request_access_token_params: typing.Optional[bool] = False,
    ):
        """
        Create new AuthorizationClient

        :param endpoint: The endpoint URL to connect to
        :param auth_endpoint: The endpoint URL where auth metadata can be found
        :param token_endpoint: The endpoint URL to retrieve token from
        :param http_session: A custom httpx.AsyncClient object to use for making HTTP requests
        :param audience: Audience parameter for Auth0 (optional)
        :param scopes: List of OAuth2 scopes to request during authentication
        :param client_id: OAuth2 client ID for authentication
        :param redirect_uri: OAuth2 redirect URI for authentication callback
        :param endpoint_metadata: EndpointMetadata object to control the rendering of the page on login successful or
            failure
        :param verify: A boolean that controls whether to verify the server's TLS certificate.
            Defaults to ``True``. When set to ``False``, requests will accept any TLS certificate
            presented by the server, and will ignore hostname mismatches and/or expired certificates,
            which will make your application vulnerable to man-in-the-middle (MitM) attacks.
            Setting verify to ``False`` may be useful during local development or testing.
        :param ca_cert_path: Path to a certificate chain file for SSL verification (optional)
        :param request_auth_code_params: Dictionary of parameters to add to login URI opened in the browser (optional)
        :param request_access_token_params: Dictionary of parameters to add when exchanging the auth code for the
            access token (optional)
        :param refresh_access_token_params: Dictionary of parameters to add when refreshing the access token (optional)
        :param add_request_auth_code_params_to_request_access_token_params: Whether to add the
            `request_auth_code_params` to the parameters sent when exchanging the auth code for the access token.
            Defaults to False. Required for the PKCE flow with the backend. Not required for the standard OAuth2 flow
                on GCP.
        """
        self._endpoint = endpoint
        self._auth_endpoint = auth_endpoint
        if endpoint_metadata is None:
            remote_url = _urlparse.urlparse(self._auth_endpoint)
            self._remote = EndpointMetadata(endpoint=str(remote_url.hostname))
        else:
            self._remote = endpoint_metadata
        self._token_endpoint = token_endpoint
        self._client_id = client_id
        self._audience = audience
        self._scopes = scopes or []
        self._redirect_uri = redirect_uri
        state = _generate_state_parameter()
        self._state = state
        self._verify = verify
        self._ca_cert_path = ca_cert_path
        self._headers = {"content-type": "application/x-www-form-urlencoded"}
        self._lock = threading.Lock()
        self._cached_credentials: typing.Optional[Credentials] = None
        self._cached_credentials_ts: float | None = None
        self._http_session = http_session

        self._request_auth_code_params = {
            "client_id": client_id,  # This must match the Client ID of the OAuth application.
            "response_type": "code",  # Indicates the authorization code grant
            "scope": " ".join(s.strip("' ") for s in self._scopes).strip(
                "[]'"
            ),  # ensures that the /token endpoint returns an ID and refresh token
            # callback location where the user-agent will be directed to.
            "redirect_uri": self._redirect_uri,
            "state": state,
        }

        # Conditionally add audience param if provided - value is not None
        if self._audience:
            self._request_auth_code_params["audience"] = self._audience

        if request_auth_code_params:
            # Allow adding additional parameters to the request_auth_code_params
            self._request_auth_code_params.update(request_auth_code_params)

        self._request_access_token_params = request_access_token_params or {}
        self._refresh_access_token_params = refresh_access_token_params or {}

        if add_request_auth_code_params_to_request_access_token_params:
            self._request_access_token_params.update(self._request_auth_code_params)

    def __repr__(self):
        return (
            f"AuthorizationClient({self._auth_endpoint}, {self._token_endpoint}, {self._client_id}, {self._scopes},"
            f" {self._redirect_uri})"
        )

    async def _create_callback_server(self):
        server_url = _urlparse.urlparse(self._redirect_uri)
        server_address = (server_url.hostname, server_url.port)
        queue = Queue()
        handler = OAuthCallbackHandler(queue, self._remote, server_url.path)
        server = await asyncio.start_server(handler.handle, server_address[0], server_address[1])
        return server, queue, handler

    async def _request_authorization_code(self):
        scheme, netloc, path, _, _, _ = _urlparse.urlparse(self._auth_endpoint)
        query = _urlencode(self._request_auth_code_params)
        endpoint = _urlparse.urlunparse((scheme, netloc, path, None, query, None))
        logger.debug(f"Requesting authorization code through {endpoint}")

        success = webbrowser.open_new_tab(endpoint)  # type: ignore
        if not success:
            click.secho(f"Please open the following link in your browser to authenticate: {endpoint}")

    async def _credentials_from_response(self, auth_token_resp) -> Credentials:
        """
        Extracts credentials from the authentication token response.

        The auth_token_resp body is of the form:
        {
          "access_token": "foo",
          "refresh_token": "bar",
          "token_type": "Bearer"
        }

        Can additionally contain "expires_in" and "id_token" fields.

        :param auth_token_resp: The HTTP response containing the token information
        :return: Credentials object created from the response
        :raises ValueError: If the response does not contain an access token
        """
        response_body = auth_token_resp.json()
        refresh_token = None
        expires_in = None

        if "access_token" not in response_body:
            raise ValueError('Expected "access_token" in response from oauth server')
        if "refresh_token" in response_body:
            refresh_token = response_body["refresh_token"]
        if "expires_in" in response_body:
            expires_in = response_body["expires_in"]
        access_token = response_body["access_token"]

        return Credentials(
            access_token=access_token,
            refresh_token=refresh_token,
            for_endpoint=self._endpoint,
            expires_in=expires_in,
        )

    async def _request_access_token(self, auth_code) -> Credentials:
        if self._state != auth_code.state:
            raise ValueError(f"Unexpected state parameter [{auth_code.state}] passed")

        params = {
            "code": auth_code.code,
            "grant_type": "authorization_code",
        }

        params.update(self._request_access_token_params)

        resp = await self._http_session.post(
            url=self._token_endpoint,
            data=params,
            headers=self._headers,
            follow_redirects=False,
        )

        if resp.status_code != _StatusCodes.OK:
            raise RuntimeError(
                "Failed to request access token with response: [{}] {!r}".format(resp.status_code, resp.content)
            )

        return await self._credentials_from_response(resp)

    async def get_creds_from_remote(self) -> Credentials:
        """
        This is the entrypoint method. It will kickoff the full authentication
        flow and trigger a web-browser to retrieve credentials. Because this
        needs to open a port on localhost and may be called from a
        multithreaded context (e.g. pyflyte register), this call may block
        multiple threads and return a cached result for up to 60 seconds.

        :return: Credentials obtained from the authentication flow
        :raises: May raise authentication-related exceptions if the flow fails
        """
        # In the absence of globally-set token values, initiate the token request flow
        with self._lock:
            # Clear cache if it's been more than 60 seconds since the last check
            cache_ttl_s = 60
            if self._cached_credentials_ts is not None and self._cached_credentials_ts + cache_ttl_s < time.monotonic():
                self._cached_credentials = None

            if self._cached_credentials is not None:
                return self._cached_credentials

            server, queue, handler = await self._create_callback_server()
            async with server:
                await self._request_authorization_code()
                # Wait for the callback handler to receive a response instead of serving forever
                await handler.response_received.wait()

            auth_code = queue.get()
            self._cached_credentials = await self._request_access_token(auth_code)
            self._cached_credentials_ts = time.monotonic()
            return self._cached_credentials

    async def refresh_access_token(self, credentials: Credentials) -> Credentials:
        """
        Refreshes the access token using the refresh token from the provided credentials.

        :param credentials: The credentials containing the refresh token to use
        :return: Updated credentials with a new access token
        :raises AccessTokenNotFoundError: If no refresh token is available in the credentials
        """
        if credentials.refresh_token is None:
            raise AccessTokenNotFoundError("no refresh token available with which to refresh authorization credentials")

        data = {
            "refresh_token": credentials.refresh_token,
            "grant_type": "refresh_token",
            "client_id": self._client_id,
        }

        data.update(self._refresh_access_token_params)

        async with typing.cast(
            typing.AsyncContextManager[Response],
            self._http_session.post(
                url=self._token_endpoint,
                data=data,
                headers=self._headers,
                follow_redirects=False,
            ),
        ) as resp:
            if resp.status_code != _StatusCodes.OK:
                raise AccessTokenNotFoundError(f"Non-200 returned from refresh token endpoint {resp.status_code}")

            return await self._credentials_from_response(resp)


class OAuthCallbackHandler:
    """
    Handles OAuth2 callback requests during the authentication flow.

    This class implements an HTTP request handler that processes the callback from the OAuth2 provider,
    extracts the authorization code, and passes it to the authentication flow.
    """

    def __init__(self, queue: Queue, remote_metadata: EndpointMetadata, redirect_path: str):
        """
        Initialize the OAuth callback handler.

        :param queue: Queue to put the authorization code into when received
        :param remote_metadata: Metadata about the remote endpoint for rendering success/failure pages
        :param redirect_path: The path component of the redirect URI to match incoming requests against
        """
        self.queue = queue
        self.remote_metadata = remote_metadata
        self.redirect_path = redirect_path
        self.response_received = asyncio.Event()

    async def handle(self, reader, writer):
        """
        Handles an incoming HTTP request during the OAuth2 callback.

        This method reads the incoming HTTP request, parses it, and if it matches the expected redirect path,
        extracts the authorization code and state from the query parameters and puts them in the queue.
        It then responds with an appropriate HTTP response.

        :param reader: The StreamReader for reading the incoming request
        :param writer: The StreamWriter for writing the response
        """
        data = await reader.read(1024)
        message = data.decode()
        headers = message.split("\r\n")
        path = headers[0].split(" ")[1]
        url = _urlparse.urlparse(path)
        if url.path.strip("/") == self.redirect_path.strip("/"):
            response = f"HTTP/1.1 {_StatusCodes.OK.value} {_StatusCodes.OK.phrase}\r\n"
            response += "Content-Type: text/html\r\n\r\n"
            self.handle_login(dict(_urlparse.parse_qsl(url.query)))
            if self.remote_metadata.success_html is None:
                response += get_default_success_html(self.remote_metadata.endpoint)
            writer.write(response.encode(_utf_8))
            await writer.drain()
        else:
            response = f"HTTP/1.1 {_StatusCodes.NOT_FOUND.value} {_StatusCodes.NOT_FOUND.phrase}\r\n\r\n"
            writer.write(response.encode(_utf_8))
            await writer.drain()
        writer.close()
        # Signal that we've received a response
        self.response_received.set()

    def handle_login(self, data: dict):
        """
        Processes the login data from the OAuth2 callback.

        Extracts the authorization code and state from the query parameters and puts them in the queue
        for the authentication flow to process.

        :param data: Dictionary containing the query parameters from the callback URL
        """
        self.queue.put(AuthorizationCode(code=data["code"], state=data["state"]))


class EndpointMetadata(pydantic.BaseModel):
    """
    This class can be used to control the rendering of the page on login successful or failure.

    :param endpoint: The endpoint URL or hostname for the remote service
    :param success_html: Optional HTML content to display on successful authentication
    :param failure_html: Optional HTML content to display on authentication failure
    """

    endpoint: str
    success_html: typing.Optional[bytes] = None
    failure_html: typing.Optional[bytes] = None


class AuthorizationCode(pydantic.BaseModel):
    """
    Represents an authorization code received from the OAuth2 provider.

    :param code: The authorization code received from the OAuth2 provider
    :param state: The state parameter that was sent in the original request
    """

    code: str
    state: str


async def _create_code_challenge(code_verifier):
    """
    Creates a code challenge for PKCE flow from the provided code verifier.
    Adapted from https://github.com/openstack/deb-python-oauth2client/blob/master/oauth2client/_pkce.py.

    :param str code_verifier: A code verifier string generated by _generate_code_verifier()
    :return str: Urlsafe base64-encoded sha256 hash digest of the code verifier
    """
    code_challenge = hashlib.sha256(code_verifier.encode(_utf_8)).digest()
    code_challenge = base64.urlsafe_b64encode(code_challenge).decode(_utf_8)
    # Eliminate invalid characters
    code_challenge = code_challenge.replace("=", "")
    return code_challenge


def _generate_state_parameter():
    """
    Generates a random state parameter for OAuth2 authorization requests.

    The state parameter is used to maintain state between the request and callback
    and to prevent cross-site request forgery attacks.

    :return: A random string to use as the state parameter
    """
    state = base64.urlsafe_b64encode(os.urandom(_random_seed_length)).decode(_utf_8)
    # Eliminate invalid characters.
    code_verifier = re.sub("[^a-zA-Z0-9-_.,]+", "", state)
    return code_verifier


async def _generate_code_verifier():
    """
    Generates a 'code_verifier' for PKCE OAuth2 flow as described in RFC 7636 section 4.1.
    Adapted from https://github.com/openstack/deb-python-oauth2client/blob/master/oauth2client/_pkce.py.

    :return str: A random string to use as the code verifier
    """
    code_verifier = base64.urlsafe_b64encode(os.urandom(_code_verifier_length)).decode(_utf_8)
    # Eliminate invalid characters.
    code_verifier = re.sub(r"[^a-zA-Z0-9_\-.~]+", "", code_verifier)
    if len(code_verifier) < 43:
        raise ValueError("Verifier too short. number of bytes must be > 30.")
    elif len(code_verifier) > 128:
        raise ValueError("Verifier too long. number of bytes must be < 97.")
    return code_verifier
