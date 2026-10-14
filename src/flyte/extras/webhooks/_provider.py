"""The contract every `flyteplugins-webhooks-<product>` plugin implements.

A provider plugin is small on purpose: say which header carries the credential,
how to verify it, and how to turn a payload into a `WebhookEvent`. Everything
else — the app, the dashboard, dispatch, the allowlist, idempotent launching —
lives here in core and is shared.

Verification helpers live here too rather than in each plugin, because they are
the part that is easy to get subtly wrong and expensive to get wrong six times.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Mapping

from ._errors import SignatureError
from ._event import WebhookEvent

#: Verify a delivery: `(raw body, headers, secret) -> bool`. Must never raise on
#: attacker-controlled input — return False instead.
VerifyFn = Callable[[bytes, Mapping[str, str], str], bool]

#: Turn a verified delivery into an event: `(headers, raw body) -> WebhookEvent`.
ParseFn = Callable[[Mapping[str, str], bytes], WebhookEvent]

#: Answer a provider's setup handshake before events flow, if it has one:
#: `(headers, raw body) -> response dict or None`. Runs *before* verification,
#: so keep it to echoing a challenge.
HandshakeFn = Callable[[Mapping[str, str], bytes], "dict[str, Any] | None"]


@dataclass(frozen=True)
class Provider:
    """Everything core needs to accept one product's webhooks.

    Args:
        name: URL segment and `WebhookEvent.provider` value. Must match the
            plugin's module name, so `/webhook/github` maps to
            `flyteplugins.github`.
        secret_env: Environment variable holding the signing secret or shared
            token. `WebhookAppEnvironment` mounts a `flyte.Secret` for it
            automatically, so it rarely needs naming twice. Defaults to the
            subclass's `default_secret_env`.
        verify: Returns True when a delivery is authentic.
        parse: Turns a verified delivery into a `WebhookEvent`.
        handshake: Optional setup handshake, answered before verification.
        signed: False when the product does not sign its webhooks at all, so the
            dashboard can say so rather than implying a guarantee that is absent.
        setup_hint: Where to configure the webhook, shown on the dashboard.
    """

    #: The environment variable this provider reads its secret from unless told
    #: otherwise. Subclasses set it, `WebhookAppEnvironment` mounts it, and
    #: `secret_env` on an instance is what actually takes effect.
    default_secret_env: ClassVar[str] = ""

    name: str
    secret_env: str
    verify: VerifyFn
    parse: ParseFn
    handshake: HandshakeFn | None = None
    signed: bool = True
    setup_hint: str = ""


# ----------------------------------------------------------------------
# verification helpers, shared so the tricky parts are written once
# ----------------------------------------------------------------------


def constant_time_equals(a: str, b: str) -> bool:
    """Compare two credentials in constant time, without raising.

    Always compares bytes. `hmac.compare_digest` raises `TypeError` on `str`
    operands containing non-ASCII, and these values come off the wire — ASGI
    servers hand Starlette raw header bytes, which it decodes as latin-1, so a
    crafted header would otherwise turn a clean 401 into a 500.
    """
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def hex_hmac_sha256(secret: str, body: bytes) -> str:
    """Hex HMAC-SHA256 of the raw body — the scheme most products use."""
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def lower_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Lowercase header keys, since HTTP header names are case-insensitive."""
    return {k.lower(): v for k, v in headers.items()}


def json_body(body: bytes) -> dict[str, Any]:
    """Parse a JSON body into a dict, raising `SignatureError` when it is not one."""
    try:
        data = json.loads(body.decode("utf-8")) if body else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SignatureError(f"invalid webhook body: {exc}") from exc
    return data if isinstance(data, dict) else {"data": data}
