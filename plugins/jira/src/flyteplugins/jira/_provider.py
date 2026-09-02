"""Jira webhook verification and payload normalization.

Jira Cloud does **not** sign its webhooks. There is no HMAC to check, so this
plugin authenticates with a shared token in `X-Webhook-Token` — which something
in front of the app has to inject, because Jira itself cannot send custom
headers. `JiraProvider` reports `signed=False` so the dashboard says so plainly rather than
implying a guarantee that is not there.
"""

from __future__ import annotations

from typing import Mapping

from flyte.extras.webhooks import (
    Provider,
    WebhookEvent,
    constant_time_equals,
    json_body,
    lower_headers,
)

#: Environment variable this provider reads its secret from by default.
DEFAULT_SECRET_ENV = "JIRA_WEBHOOK_TOKEN"


def verify(body: bytes, headers: Mapping[str, str], secret: str) -> bool:
    """Compare the `X-Webhook-Token` header against the shared token."""
    token = lower_headers(headers).get("x-webhook-token")
    if not token:
        return False
    return constant_time_equals(token.strip(), secret)


def parse(headers: Mapping[str, str], body: bytes) -> WebhookEvent:
    """Normalize a Jira delivery into a `WebhookEvent`."""
    payload = json_body(body)
    issue = payload.get("issue") or {}
    fields = issue.get("fields") or {}
    user = payload.get("user") or {}
    return WebhookEvent(
        provider="jira",
        event_type=payload.get("webhookEvent", "unknown"),
        delivery_id=str(payload.get("timestamp") or ""),
        resource_id=issue.get("key"),
        occurred_at=str(payload.get("timestamp")) if payload.get("timestamp") is not None else None,
        scope=(fields.get("project") or {}).get("key"),
        title=fields.get("summary"),
        actor=user.get("displayName") or user.get("name"),
        payload=payload,
    )


class JiraProvider(Provider):
    """Jira's webhook provider, with its defaults pre-wired.

    ```python
    from flyte.extras.webhooks import WebhookAppEnvironment
    from flyteplugins.jira import JiraProvider

    app_env = WebhookAppEnvironment(name="webhooks", providers=[JiraProvider()])
    ```

    Jira does not sign its webhooks, so this provider authenticates with a
    shared token instead and reports `signed=False` — which is what makes the
    dashboard say so rather than implying a guarantee that is absent.

    Args:
        secret_env: Environment variable holding the secret, mounted from a
            `flyte.Secret`. Override only if you store it under a non-standard
            name; the default is what the docs and examples assume.
    """

    def __init__(self, *, secret_env: str = DEFAULT_SECRET_ENV) -> None:
        super().__init__(
            name="jira",
            secret_env=secret_env,
            verify=verify,
            parse=parse,
            signed=False,
            setup_hint="Jira Settings -> System -> Webhooks (needs a proxy to inject X-Webhook-Token)",
        )
