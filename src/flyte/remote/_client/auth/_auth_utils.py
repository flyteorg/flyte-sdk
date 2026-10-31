from __future__ import annotations

import base64
import binascii
from typing import Literal


def decode_api_key(encoded_str: str) -> tuple[str, str, str, str | Literal["None"]]:
    """Decode encoded base64 string into app credentials. endpoint, client_id, client_secret, org"""
    from flyte.errors import InitializationError

    # Split with maxsplit=3 to handle endpoints with colons (e.g., dns:///endpoint.com)
    try:
        decoded = base64.b64decode(encoded_str.encode("utf-8")).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError) as e:
        # A malformed/garbled API key (e.g. truncated or hand-edited) is user input, not an
        # SDK bug. Raise a typed user error so it surfaces with a clear message and is filtered
        # from crash reporting instead of leaking as a raw base64/binascii error (FLYTE-SDK-5C).
        raise InitializationError(
            "InvalidApiKey",
            "user",
            "Invalid API key: the value is not a valid base64-encoded key. "
            "Re-copy the API key from the Union console and try again.",
        ) from e

    parts = decoded.split(":", 3)
    if len(parts) != 4:
        raise InitializationError(
            "InvalidApiKey",
            "user",
            f"Invalid API key format. Expected 4 parts separated by ':', got {len(parts)}. "
            "Re-copy the API key from the Union console and try again.",
        )

    endpoint, client_id, client_secret, org = parts
    # For consistency, let's make sure org is always a non-empty string
    if not org:
        org = "None"

    return endpoint, client_id, client_secret, org
