from flyte._logging import logger
from flyte.remote._client.auth import _token_client as token_client
from flyte.remote._client.auth._authenticators.base import Authenticator
from flyte.remote._client.auth._keyring import Credentials


class ClientCredentialsAuthenticator(Authenticator):
    """
    This Authenticator uses ClientId and ClientSecret to authenticate
    """

    def __init__(
        self,
        client_id: str,
        client_credentials_secret: str,
        **kwargs,
    ):
        """
        Initialize the client credentials authenticator.

        Args:
            client_id: The client ID for authentication
            client_credentials_secret: The client secret for authentication
            kwargs: Additional keyword arguments passed to the base Authenticator
                **Keyword Arguments passed to base Authenticator**:
            endpoint: The endpoint URL for authentication (required)
            cfg_store: Optional client configuration store for retrieving remote configuration
            client_config: Optional client configuration containing authentication settings
            credentials: Optional credentials to use for authentication
            http_session: Optional HTTP session to use for requests
            http_proxy_url: Optional HTTP proxy URL
            verify: Whether to verify SSL certificates (default: True)
            ca_cert_path: Optional path to CA certificate file
            scopes: List of scopes to request during authentication
            audience: Audience for the token
        """
        if not client_id or not client_credentials_secret:
            raise ValueError("both client_id and client_credentials_secret are required.")
        self._client_id = client_id
        self._client_credentials_secret = client_credentials_secret
        super().__init__(**kwargs)

    async def _do_refresh_credentials(self) -> Credentials:
        """
        Refreshes the authentication credentials using client credentials flow.

        This function is used by the _handle_rpc_error() decorator, depending on the AUTH_MODE config object.
        This handler is meant for SDK use-cases of auth (like pyflyte, or when users call SDK functions that require
        access to Admin, like when waiting for another workflow to complete from within a task). This function uses
        basic auth, which means the credentials for basic auth must be present from wherever this code is running.
        """
        cfg = await self._resolve_config()

        # Note that unlike the Pkce flow, the client ID does not come from Admin.
        logger.debug(f"Basic authorization flow with client id {self._client_id} scope {cfg.scopes}")
        authorization_header = token_client.get_basic_authorization_header(
            self._client_id, self._client_credentials_secret
        )

        token, refresh_token, expires_in = await token_client.get_token(
            token_endpoint=cfg.token_endpoint,
            authorization_header=authorization_header,
            http_proxy_url=self._http_proxy_url,
            verify=self._verify,
            scopes=cfg.scopes,
            audience=cfg.audience,
            http_session=self._http_session,
        )

        logger.info("Retrieved new token, expires in {}".format(expires_in))
        return Credentials(
            for_endpoint=self._endpoint, access_token=token, refresh_token=refresh_token, expires_in=expires_in
        )
