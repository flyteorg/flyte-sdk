import asyncio
import os
import time
import typing
from pathlib import Path

import httpx

from flyte._logging import logger
from flyte.remote._client.auth._authenticators.base import Authenticator, AuthHeaders
from flyte.remote._client.auth._keyring import Credentials
from flyte.remote._client.auth.errors import AuthenticationError

# Set on task pods by the platform when the dataplane STS is enabled.
STS_ENDPOINT_ENV_VAR = "_U_STS_ENDPOINT"
STS_TOKEN_FILE_ENV_VAR = "_U_STS_TOKEN_FILE"

_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"
_JWT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:jwt"
# Refresh once this share of a token's life has passed, so a request never
# starts with a token about to expire.
_REFRESH_AT = 0.8


class WorkloadTokenAuthenticator(Authenticator):
    """
    Authenticates a task pod as the creator of its run, without any long-lived secret in the pod.

    The pod's projected service-account token (rotated by the kubelet, re-read on every refresh) is
    exchanged at the dataplane STS (RFC 8693 token exchange) for a short-lived token the control
    plane accepts for this run's project and domain only.
    """

    def __init__(
        self,
        sts_endpoint: str | None = None,
        token_file: str | None = None,
        sts_http_session: httpx.AsyncClient | None = None,
        **kwargs,
    ):
        self._sts_endpoint: str = sts_endpoint or os.getenv(STS_ENDPOINT_ENV_VAR) or ""
        self._token_file: str = token_file or os.getenv(STS_TOKEN_FILE_ENV_VAR) or ""
        if not self._sts_endpoint or not self._token_file:
            raise AuthenticationError(
                f"WorkloadToken auth needs {STS_ENDPOINT_ENV_VAR} and {STS_TOKEN_FILE_ENV_VAR}; is the STS enabled?"
            )
        super().__init__(**kwargs)
        # The STS is an in-cluster endpoint; it never goes through a proxy configured for the control plane.
        self._sts_session = sts_http_session or httpx.AsyncClient(
            timeout=10.0, verify=self._verify if self._verify is not None else True
        )
        self._obtained_at = 0.0

    async def _do_refresh_credentials(self) -> Credentials:
        try:
            # Read on every refresh: the kubelet rotates the projected token in place.
            subject_token = (await asyncio.to_thread(Path(self._token_file).read_text)).strip()
        except OSError as e:
            raise AuthenticationError(f"Cannot read the projected token at {self._token_file}: {e}") from e
        if not subject_token:
            raise AuthenticationError(f"The projected token at {self._token_file} is empty")

        try:
            resp = await self._sts_session.post(
                self._sts_endpoint,
                data={
                    "grant_type": _GRANT_TYPE,
                    "subject_token_type": _JWT_TOKEN_TYPE,
                    "subject_token": subject_token,
                },
                headers={"Accept": "application/json"},
            )
        except httpx.HTTPError as e:
            raise AuthenticationError(f"The STS at {self._sts_endpoint} is unreachable: {e}") from e
        if resp.status_code != 200:
            error = ""
            try:
                error = resp.json().get("error", "")
            except ValueError:
                pass
            raise AuthenticationError(f"The STS refused the token exchange ({resp.status_code} {error})".strip())
        body = resp.json()
        token = body.get("access_token")
        if not token or body.get("token_type", "").lower() != "bearer":
            raise AuthenticationError("The STS answered without a bearer token")
        expires_in = body.get("expires_in")
        self._obtained_at = time.monotonic()
        logger.debug(f"Obtained a workload token, expires in {expires_in}s")
        return Credentials(
            for_endpoint=self._endpoint,
            access_token=token,
            expires_in=int(expires_in) if expires_in is not None else None,
        )

    async def get_auth_headers(self) -> typing.Optional[AuthHeaders]:
        creds = self.get_credentials()
        if creds is not None and creds.expires_in:
            if time.monotonic() - self._obtained_at >= creds.expires_in * _REFRESH_AT:
                await self.refresh_credentials(creds.id)
        return await super().get_auth_headers()
