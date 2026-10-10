import urllib.parse

from rich import print as rich_print

from flyte._logging import logger
from flyte.remote._client.auth import _token_client as token_client
from flyte.remote._client.auth._authenticators.base import Authenticator
from flyte.remote._client.auth._keyring import Credentials
from flyte.remote._client.auth.errors import AuthenticationError, AuthenticationPending


def _verification_uri_with_code(verification_uri: str, user_code: str) -> str:
    """
    Add `user_code` to `verification_uri` as a query parameter.

    Used only when the server did not send `verification_uri_complete`. The URI
    is parsed and rebuilt rather than concatenated, so a verification_uri that
    already carries a query (`.../device?tenant=x`) gains `&user_code=...`
    instead of a second `?`, and a fragment stays after the query.
    """
    parts = urllib.parse.urlsplit(verification_uri)
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query.append(("user_code", user_code))
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


class DeviceCodeAuthenticator(Authenticator):
    """
    This Authenticator implements the Device Code authorization flow useful for headless user authentication.

    Examples described
    - https://developer.okta.com/docs/guides/device-authorization-grant/main/
    - https://auth0.com/docs/get-started/authentication-and-authorization-flow/device-authorization-flow#device-flow
    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialize the device code authenticator.

        Args:
            kwargs: Keyword arguments passed to the base Authenticator
                **Keyword Arguments passed to base Authenticator**:
            endpoint: The endpoint URL for authentication (required)
            cfg_store: Optional client configuration store for retrieving remote configuration
            client_config: Optional client configuration containing authentication settings
            credentials: Optional credentials to use for authentication
            http_session: Optional HTTP session to use for requests
            http_proxy_url: Optional HTTP proxy URL
            verify: Whether to verify SSL certificates (default: True)
            ca_cert_path: Optional path to CA certificate file
            client_id: Client ID for authentication
            scopes: List of scopes to request during authentication
            audience: Audience for the token
            device_authorization_endpoint: Endpoint for device authorization
        """

        super().__init__(
            **kwargs,
        )

    async def _do_refresh_credentials(self) -> Credentials:
        """
        Refreshes the authentication credentials using device code flow.

        First attempts to refresh using a refresh token if available.
        If that fails, falls back to the full device code authorization flow.
        """
        cfg = await self._resolve_config()

        # These always come from the public client config. The remote config store fills this in
        # from the OAuth2 metadata proto, where an unadvertised endpoint arrives as "" rather than
        # None -- so an `is None` check let the empty string through and `httpx.post("")` blew up
        # with `UnsupportedProtocol: Request URL is missing an 'http://' or 'https://' protocol.`
        # instead of the actionable message below (FLYTE-SDK-6P). Treat empty as absent, matching
        # `ClientConfig.merge`, which already uses `or` on this field.
        if not cfg.device_authorization_endpoint:
            raise AuthenticationError(
                "Device Authentication is not available on the Flyte backend / authentication server"
            )

        if self._creds and self._creds.refresh_token:
            """We have an refresh token so lets try to refresh it"""
            try:
                access_token, refresh_token, expires_in = await token_client.get_token(
                    token_endpoint=cfg.token_endpoint,
                    client_id=cfg.client_id,
                    audience=cfg.audience,
                    scopes=cfg.scopes,
                    http_proxy_url=self._http_proxy_url,
                    verify=self._verify,
                    grant_type=token_client.GrantType.REFRESH_TOKEN,
                    refresh_token=self._creds.refresh_token,
                    http_session=self._http_session,
                )

                return Credentials(
                    access_token=access_token,
                    refresh_token=refresh_token,
                    expires_in=expires_in,
                    for_endpoint=self._endpoint,
                )
            except (AuthenticationError, AuthenticationPending):
                logger.warning("Logging in...")

        """Fall back to device flow"""
        resp = await token_client.get_device_code(
            cfg.device_authorization_endpoint,
            cfg.client_id,
            audience=cfg.audience,
            scopes=cfg.scopes,
            http_session=self._http_session,
        )

        # Prefer the server's own pre-filled URI. Falling back means building it
        # by hand, and that has to merge into any query verification_uri already
        # carries rather than appending a second "?" -- and leave a fragment where
        # it belongs, after the query.
        full_uri = resp.verification_uri_complete or _verification_uri_with_code(resp.verification_uri, resp.user_code)
        text = (
            f"To Authenticate, navigate in a browser to the following URL: [blue link={full_uri}]{full_uri}[/blue link]"
        )
        rich_print(text)
        try:
            token, refresh_token, expires_in = await token_client.poll_token_endpoint(
                resp,
                token_endpoint=cfg.token_endpoint,
                client_id=cfg.client_id,
                audience=cfg.audience,
                scopes=cfg.scopes,
                http_proxy_url=self._http_proxy_url,
                verify=self._verify,
                http_session=self._http_session,
            )

            return Credentials(
                access_token=token, refresh_token=refresh_token, expires_in=expires_in, for_endpoint=self._endpoint
            )

        except Exception:
            raise
