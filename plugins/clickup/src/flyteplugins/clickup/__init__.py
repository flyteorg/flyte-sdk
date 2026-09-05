"""ClickUp webhooks for Flyte.

Hand a `ClickUpProvider()` to a `WebhookAppEnvironment` and register handlers with the
typed constants in `events`. Calling the ClickUp API is not this plugin's job —
it is a handful of REST calls, so use `httpx` from your tasks. See
`examples/external_saas_integrations`.
"""

import hashlib
import hmac

from . import events
from ._provider import ClickUpProvider, parse, verify

__all__ = ["SAMPLE_DELIVERY", "ClickUpProvider", "events", "parse", "verify"]


def _sample_headers(body: bytes, secret: str) -> dict[str, str]:
    return {"X-Clickup-Signature": hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()}


#: A real `taskCreated` delivery, trimmed to the fields the parser reads.
SAMPLE_DELIVERY = (
    _sample_headers,
    (
        b'{"event": "taskCreated", "task_id": "abc123", "list_id": "9000",'
        b' "webhook_id": "wh-000", "timestamp": 1700000000000,'
        b' "task": {"id": "abc123", "name": "Fix the thing",'
        b' "url": "https://app.clickup.com/t/abc123"}}'
    ),
)
