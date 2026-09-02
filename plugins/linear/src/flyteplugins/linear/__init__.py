"""Linear webhooks for Flyte.

Hand a `LinearProvider()` to a `WebhookAppEnvironment` and register handlers with the
typed constants in `events`. Calling the Linear API is not this plugin's job —
Linear's API is a single GraphQL endpoint, so use `gql` from your tasks. See
`examples/external_saas_integrations`.
"""

import hashlib
import hmac

from . import events
from ._provider import DEFAULT_SECRET_ENV, LinearProvider, parse, verify

__all__ = ["DEFAULT_SECRET_ENV", "SAMPLE_DELIVERY", "LinearProvider", "events", "parse", "verify"]


def _sample_headers(body: bytes, secret: str) -> dict[str, str]:
    return {"X-Linear-Signature": hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()}


#: A real `Issue.create` delivery, trimmed to the fields the parser reads.
SAMPLE_DELIVERY = (
    _sample_headers,
    (
        b'{"action": "create", "type": "Issue", "webhookId": "wh-000",'
        b' "createdAt": "2024-01-01T00:00:00.000Z",'
        b' "data": {"id": "00000000-0000-0000-0000-000000000000", "title": "A bug",'
        b' "teamId": "team-000", "updatedAt": "2024-01-01T00:00:00.000Z",'
        b' "url": "https://linear.app/acme/issue/ENG-1"}}'
    ),
)
