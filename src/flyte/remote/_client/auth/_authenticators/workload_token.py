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
        creds = self.get_credentials()
        if creds is None or time.monotonic() - self._read_at >= _REREAD_AFTER:
            await self.refresh_credentials(creds.id if creds is not None else None)
        headers = await super().get_auth_headers()
        if headers is not None:
            headers.headers[WORKLOAD_HEADER] = self._workload
        return headers
