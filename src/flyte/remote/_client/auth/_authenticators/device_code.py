from rich import print as rich_print

from flyte._logging import logger
from flyte.remote._client.auth import _token_client as token_client
from flyte.remote._client.auth._authenticators.base import Authenticator
from flyte.remote._client.auth._keyring import Credentials
from flyte.remote._client.auth.errors import AuthenticationError, AuthenticationPending


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

        full_uri = f"{resp.verification_uri}?user_code={resp.user_code}"
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
