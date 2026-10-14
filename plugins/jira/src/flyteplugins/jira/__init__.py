"""Jira webhooks for Flyte.

Hand a `JiraProvider()` to a `WebhookAppEnvironment` and register handlers with the
typed constants in `events`. Calling the Jira API is not this plugin's job — use
the `jira` package from your tasks. See `examples/external_saas_integrations`.

Note Jira does not sign its webhooks; see `_provider` for what this plugin does
instead.
"""

from . import events
from ._provider import JiraProvider, parse, verify

__all__ = ["SAMPLE_DELIVERY", "JiraProvider", "events", "parse", "verify"]


def _sample_headers(body: bytes, secret: str) -> dict[str, str]:
    # No signature to compute: Jira sends a static shared token.
    return {"X-Webhook-Token": secret}


#: A real `jira:issue_created` delivery, trimmed to the fields the parser reads.
SAMPLE_DELIVERY = (
    _sample_headers,
    (
        b'{"webhookEvent": "jira:issue_created", "timestamp": 1700000000000,'
        b' "user": {"displayName": "Bob"},'
        b' "issue": {"key": "PROJ-1", "id": "10001",'
        b' "fields": {"summary": "A bug", "project": {"key": "PROJ"}}}}'
    ),
)
