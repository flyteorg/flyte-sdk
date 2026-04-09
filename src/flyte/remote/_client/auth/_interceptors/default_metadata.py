import os
import random
import string
from uuid import uuid4

import flyte

ALPHABET = string.ascii_lowercase + string.digits
TRUSTED_IDENTITY_CLAIM_ENV_VAR = "_U_EXTERNAL_IDENTITY_CLAIM"
TRUSTED_IDENTITY_TYPE_CLAIM_ENV_VAR = "_U_EXTERNAL_IDENTITY_TYPE_CLAIM"
USER_SUBJECT_HEADER = "x-user-subject"
USER_IDENTITY_TYPE_HEADER = "x-user-claim-identitytype"


def fast_short_id():
    return "".join(random.choices(ALPHABET, k=4))


def _generate_request_id() -> str:
    """
    Generate a request ID based on the current Flyte context.
    If running within a Flyte task context, creates a request ID using the action's unique_id_str method.
    Otherwise, falls back to a UUID4.
    """
    ctx = flyte.ctx()
    if ctx and ctx.action:
        return ctx.action.unique_id_str(salt=fast_short_id())
    return str(uuid4())


class DefaultMetadataInterceptor:
    """Injects default metadata into every outgoing RPC.

    Implements the connectrpc MetadataInterceptor protocol:
    - on_start(ctx) -> token
    - on_end(token, ctx, error) -> None
    """

    def __init__(self):
        self._trusted_identity_headers = _trusted_identity_headers_from_env()

    async def on_start(self, ctx) -> None:
        headers = ctx.request_headers()
        headers.setdefault("x-request-id", _generate_request_id())
        for key, value in self._trusted_identity_headers.items():
            headers.setdefault(key, value)

    async def on_end(self, token, ctx, error) -> None:
        pass


def _trusted_identity_headers_from_env() -> dict[str, str]:
    """Build trusted identity headers from pod env when self-hosted authz requires them."""
    headers = {}
    subject = os.getenv(TRUSTED_IDENTITY_CLAIM_ENV_VAR)
    identity_type = os.getenv(TRUSTED_IDENTITY_TYPE_CLAIM_ENV_VAR)

    if subject:
        headers[USER_SUBJECT_HEADER] = subject
    if identity_type:
        headers[USER_IDENTITY_TYPE_HEADER] = identity_type

    return headers
