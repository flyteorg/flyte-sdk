"""Slack webhooks (Events API) for Flyte.

Hand a `SlackProvider()` to a `WebhookAppEnvironment` and register handlers with the
typed constants in `events`. Calling the Slack API is not this plugin's job —
use `slack_sdk` from your tasks. See `examples/external_saas_integrations`.
"""

import hashlib
import hmac
import time

from . import events
from ._provider import MAX_REQUEST_AGE_SECONDS, SlackProvider, handshake, parse, verify

__all__ = [
    "MAX_REQUEST_AGE_SECONDS",
    "SAMPLE_DELIVERY",
    "SlackProvider",
    "events",
    "handshake",
    "parse",
    "verify",
]


def _sample_headers(body: bytes, secret: str) -> dict[str, str]:
    # Signed at "now" so the delivery is inside the replay window whenever
    # conformance runs.
    timestamp = str(int(time.time()))
    base = b"v0:" + timestamp.encode() + b":" + body
    signature = hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()
    return {"X-Slack-Request-Timestamp": timestamp, "X-Slack-Signature": f"v0={signature}"}


#: A real `app_mention` event callback, trimmed to the fields the parser reads.
SAMPLE_DELIVERY = (
    _sample_headers,
    (
        b'{"event_id": "Ev00000000", "team_id": "T00000000",'
        b' "event": {"type": "app_mention", "channel": "C00000000", "ts": "1700000000.000100",'
        b' "user": "U00000000", "text": "<@U0BOT> can you look at this"}}'
    ),
)
