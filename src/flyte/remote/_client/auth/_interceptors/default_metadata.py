import random
import string
from uuid import uuid4

import flyte
from flyte._logging import logger

ALPHABET = string.ascii_lowercase + string.digits


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
    """Injects x-request-id header into every outgoing RPC.

    Implements the connectrpc MetadataInterceptor protocol:
    - on_start(ctx) -> token
    - on_end(token, ctx, error) -> None
    """

    async def on_start(self, ctx) -> None:
        existing_rid = ctx.request_headers()["x-request-id"]
        if existing_rid is not None:
            return None

        rid = _generate_request_id()
        logger.debug(f"request-id: {rid}")
        ctx.request_headers()["x-request-id"] = rid
        return None

    async def on_end(self, token, ctx, error) -> None:
        pass
