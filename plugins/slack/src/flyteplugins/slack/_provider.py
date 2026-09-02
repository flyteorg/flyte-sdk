"""Slack Events API verification and payload normalization."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any, Mapping

from flyte.extras.webhooks import (
    Provider,
    SignatureError,
    WebhookEvent,
    constant_time_equals,
    json_body,
    lower_headers,
)

#: Environment variable this provider reads its secret from by default.
DEFAULT_SECRET_ENV = "SLACK_SIGNING_SECRET"

#: Reject requests whose timestamp is older than this (replay protection).
MAX_REQUEST_AGE_SECONDS = 60 * 5


def verify(body: bytes, headers: Mapping[str, str], secret: str) -> bool:
    """Verify the `X-Slack-Signature` v0 HMAC, within the replay window."""
    lowered = lower_headers(headers)
    timestamp, signature = lowered.get("x-slack-request-timestamp"), lowered.get("x-slack-signature")
    if not timestamp or not signature or not signature.startswith("v0="):
        return False
    try:
        sent_at = int(timestamp)
    except ValueError:
        return False
    if abs(time.time() - sent_at) > MAX_REQUEST_AGE_SECONDS:
        return False
    # Sign the raw bytes and the raw header. Decoding the body and re-encoding it
    # would corrupt any byte Slack signed but Python cannot decode, and running
    # the timestamp through int() would drop whatever formatting Slack signed.
    basestring = b"v0:" + timestamp.encode("utf-8") + b":" + body
    expected = "v0=" + hmac.new(secret.encode("utf-8"), basestring, hashlib.sha256).hexdigest()
    return constant_time_equals(expected, signature)


def handshake(headers: Mapping[str, str], body: bytes) -> dict[str, Any] | None:
    """Echo the `url_verification` challenge Slack sends before events flow."""
    try:
        data = json.loads(body.decode("utf-8")) if body else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(data, dict) and data.get("type") == "url_verification":
        return {"challenge": str(data.get("challenge", ""))}
    return None


def parse(headers: Mapping[str, str], body: bytes) -> WebhookEvent:
    """Normalize a Slack event callback into a `WebhookEvent`."""
    payload = json_body(body)
    event = payload.get("event") or {}
    if not isinstance(event, dict) or not event:
        raise SignatureError("event payload is missing its `event` object")
    channel, ts = event.get("channel"), event.get("ts")
    return WebhookEvent(
        provider="slack",
        event_type=event.get("type", "unknown"),
        action=event.get("subtype"),
        delivery_id=payload.get("event_id", ""),
        # Keyed per message. Collapse a whole thread onto one run by passing
        # `event.payload["event"]["thread_ts"]` as your own key instead.
        resource_id=f"{channel}:{ts}" if channel and ts else None,
        scope=channel,
        title=(event.get("text") or "")[:120] or None,
        actor=event.get("user"),
        payload=payload,
    )


class SlackProvider(Provider):
    """Slack's webhook provider, with its defaults pre-wired.

    ```python
    from flyte.extras.webhooks import WebhookAppEnvironment
    from flyteplugins.slack import SlackProvider

    app_env = WebhookAppEnvironment(name="webhooks", providers=[SlackProvider()])
    ```

    Args:
        secret_env: Environment variable holding the secret, mounted from a
            `flyte.Secret`. Override only if you store it under a non-standard
            name; the default is what the docs and examples assume.
    """

    def __init__(self, *, secret_env: str = DEFAULT_SECRET_ENV) -> None:
        super().__init__(
            name="slack",
            secret_env=secret_env,
            verify=verify,
            parse=parse,
            handshake=handshake,
            setup_hint="api.slack.com/apps -> Event Subscriptions",
        )
