from __future__ import annotations

import base64
from typing import Literal


def decode_api_key(encoded_str: str) -> tuple[str, str, str, str | Literal["None"]]:
    """Decode encoded base64 string into app credentials. endpoint, client_id, client_secret, org"""
    # Split with maxsplit=3 to handle endpoints with colons (e.g., dns:///endpoint.com)
    parts = base64.b64decode(encoded_str.encode("utf-8")).decode("utf-8").split(":", 3)
    if len(parts) != 4:
        raise ValueError(f"Invalid API key format. Expected 4 parts separated by ':', got {len(parts)}")

    endpoint, client_id, client_secret, org = parts
    # For consistency, let's make sure org is always a non-empty string
    if not org:
        org = "None"

    return endpoint, client_id, client_secret, org
