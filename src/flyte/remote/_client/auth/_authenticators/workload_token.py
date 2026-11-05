import asyncio
import os
import time
import typing
from pathlib import Path

from flyte._logging import logger
from flyte.remote._client.auth._authenticators.base import Authenticator, AuthHeaders
from flyte.remote._client.auth._keyring import Credentials
from flyte.remote._client.auth.errors import AuthenticationError

# Set on task pods by the platform when workload identity is enabled: the action attempt the pod runs
# (<org>/<project>/<domain>/<run>/<action>/<attempt>) and its projected service-account token.
WORKLOAD_ENV_VAR = "_U_WORKLOAD"
TOKEN_FILE_ENV_VAR = "_U_WORKLOAD_TOKEN_FILE"
# The header naming the attempt; the control plane accepts the token only for it.
WORKLOAD_HEADER = "x-union-workload"
# The api key the platform may still mount alongside the workload token (during the migration away from it).
_API_KEY_ENV_VARS = ("_UNION_EAGER_API_KEY", "EAGER_API_KEY")
# How long a read of the token file is used before reading it again. The kubelet rotates the token in place
# well before it expires (at 80% of its lifetime, an hour by default), so a minute is always fresh.
_REREAD_AFTER = 60.0


class WorkloadTokenAuthenticator(Authenticator):
    """
    Authenticates a task pod as the creator of its run, without any long-lived secret in the pod.

    The pod presents its own projected Kubernetes service-account token, and names the action attempt it runs. The
    control plane accepts it only from the pod running that attempt, on the cluster the attempt is leased to, and only
    for the run's project and domain.
    """

    def __init__(
        self,
        workload: str | None = None,
        token_file: str | None = None,
        fallback: typing.Optional[typing.Callable[[], Authenticator]] = None,
        **kwargs,
    ):
        self._workload: str = workload or os.getenv(WORKLOAD_ENV_VAR) or ""
        self._token_file: str = token_file or os.getenv(TOKEN_FILE_ENV_VAR) or ""
        if not self._workload or not self._token_file:
            raise AuthenticationError(
                f"WorkloadToken auth needs {WORKLOAD_ENV_VAR} and {TOKEN_FILE_ENV_VAR}; is workload identity enabled?"
            )
        super().__init__(**kwargs)
        self._read_at = 0.0
        # Builds the authenticator to fall back to (the mounted api key) if the control plane refuses the
        # workload token itself; None when there is nothing to fall back to.
        self._fallback_factory = fallback
        self._fallback: typing.Optional[Authenticator] = None
        # True while get_auth_headers re-reads the token on its own schedule, as opposed to a refresh after the
        # control plane refused a request.
        self._scheduled_refresh = False

    async def _do_refresh_credentials(self) -> Credentials:
        try:
            # Read again on every refresh: the kubelet rotates the projected token in place.
            token = (await asyncio.to_thread(Path(self._token_file).read_text)).strip()
        except OSError as e:
            raise AuthenticationError(f"Cannot read the projected token at {self._token_file}: {e}") from e
        if not token:
            raise AuthenticationError(f"The projected token at {self._token_file} is empty")
        self._read_at = time.monotonic()
        logger.debug(f"Read the workload token for {self._workload}")
        return Credentials(for_endpoint=self._endpoint, access_token=token)

    async def get_auth_headers(self) -> typing.Optional[AuthHeaders]:
        if self._fallback is not None:
            return await self._fallback.get_auth_headers()
        creds = self.get_credentials()
        if creds is None or time.monotonic() - self._read_at >= _REREAD_AFTER:
            self._scheduled_refresh = True
            try:
                await self.refresh_credentials(creds.id if creds is not None else None)
            finally:
                self._scheduled_refresh = False
        headers = await super().get_auth_headers()
        if headers is not None:
            headers.headers[WORKLOAD_HEADER] = self._workload
        return headers

    async def refresh_credentials(self, creds_id: str | None = None):
        if self._fallback is not None:
            return await self._fallback.refresh_credentials(creds_id)
        refused = None if self._scheduled_refresh else self.get_credentials()
        await super().refresh_credentials(creds_id)
        if refused is None or self._fallback_factory is None:
            return
        current = self.get_credentials()
        if current is not None and current.access_token == refused.access_token:
            # The control plane refused the token itself, not a stale read (a rotated token would differ): it does
            # not accept workload tokens here. Use the api key the platform still mounts, from now on.
            logger.warning(
                f"The control plane refused this pod's workload token for {self._workload}; "
                "falling back to the mounted api key"
            )
            fallback = self._fallback_factory()
            await fallback.refresh_credentials()
            self._fallback = fallback


def mounted_api_key_fallback(
    endpoint: str, cfg_store: typing.Any, **kwargs
) -> typing.Optional[typing.Callable[[], Authenticator]]:
    """
    Returns a factory for an authenticator on the api key the platform mounted next to the workload token, or None
    when there is none. The workload token is preferred; the key is only used if the control plane refuses it.
    """
    api_key = next((v for v in (os.getenv(name) for name in _API_KEY_ENV_VARS) if v), None)
    if not api_key:
        return None

    def build() -> Authenticator:
        from flyte.remote._client.auth._auth_utils import decode_api_key
        from flyte.remote._client.auth._authenticators.client_credentials import ClientCredentialsAuthenticator

        _, client_id, client_secret, _ = decode_api_key(api_key)
        rest = {k: v for k, v in kwargs.items() if k not in ("client_id", "client_secret", "client_credentials_secret")}
        return ClientCredentialsAuthenticator(
            endpoint=endpoint,
            cfg_store=cfg_store,
            client_id=client_id,
            client_secret=client_secret,
            client_credentials_secret=client_secret,
            **rest,
        )

    return build
