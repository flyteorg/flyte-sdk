"""
Per-workload credential for the host Apple Silicon bridge. The backend mints it when it builds an Apple Silicon
workload's pod (flyteplugins `flytek8s.MintAppleSiliconToken`) and the bridge verifies it, so a container can only reach
the daemon started for its own workload. Both sides share the devbox's bridge secret; the format must stay in sync with
the Go implementation.
"""

from __future__ import annotations

import hashlib
import hmac

TOKEN_VERSION = "as1"


def _mac(secret: str, workload_id: str) -> str:
    return hmac.new(secret.encode(), workload_id.encode(), hashlib.sha256).hexdigest()


def mint_token(secret: str, workload_id: str) -> str:
    """Return the credential for `workload_id`: `as1.<workload_id>.<hex hmac-sha256>`."""
    return f"{TOKEN_VERSION}.{workload_id}.{_mac(secret, workload_id)}"


def verify_token(secret: str, token: str) -> str | None:
    """Return the workload id `token` was minted for, or None if it was not minted with `secret`."""
    version, _, rest = token.partition(".")
    workload_id, _, mac = rest.rpartition(".")
    if version != TOKEN_VERSION or not workload_id or not mac:
        return None
    if not hmac.compare_digest(mac, _mac(secret, workload_id)):
        return None
    return workload_id
