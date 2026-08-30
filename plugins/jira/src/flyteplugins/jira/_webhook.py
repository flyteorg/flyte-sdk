"""Jira webhook token verification and event normalization.

Jira Cloud webhooks are *not* cryptographically signed, so the receiver
protects itself with a shared secret token: the webhook sender must be
configured to include it as the `X-Webhook-Token` header (any webhook
forwarder or proxy can inject it, and the dashboard instructions show how).

Payloads carry a `webhookEvent` name (`jira:issue_created`,
`jira:issue_updated`, ...) and the affected issue.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any, Mapping

from pydantic import BaseModel, Field

from ._client import _simplify_issue
from ._errors import WebhookSignatureError

TOKEN_HEADER = "X-Webhook-Token"


class JiraEvent(BaseModel):
    """A normalized Jira webhook event."""

    webhook_event: str
    issue_key: str | None = None
    issue_id: str | None = None
    summary: str | None = None
    status: str | None = None
    issue_type: str | None = None
    project_key: str | None = None
    actor: str | None = None
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = Field(default_factory=dict)

    @property
    def qualified_type(self) -> str:
        """The Jira webhook event name, e.g. `jira:issue_created`."""
        return self.webhook_event

    def dedupe_key(self) -> str:
        """Stable key for idempotent run launching.

        Keyed on event name + issue key + actor + transition timestamp where
        available, so redeliveries of the same event dedupe while later
        updates to the same issue produce distinct keys.
        """
        timestamp = self.payload.get("timestamp") or self.payload.get("issue_event_type_name") or ""
        base = f"{self.webhook_event}:{self.issue_key}:{timestamp}"
        return hashlib.sha256(base.encode()).hexdigest()[:32]


def verify_webhook_token(token_header: str | None, secret: str) -> bool:
    """Compare the `X-Webhook-Token` header against the shared secret."""
    if not token_header:
        return False
    return hmac.compare_digest(token_header.strip(), secret)


def parse_webhook(headers: Mapping[str, str], body: bytes) -> JiraEvent:
    """Parse webhook headers and body into a `JiraEvent`.

    Raises `WebhookSignatureError` when the body is not valid JSON.
    """
    try:
        payload = json.loads(body.decode("utf-8")) if body else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WebhookSignatureError(f"invalid webhook body: {exc}") from exc
    if not isinstance(payload, dict):
        payload = {"webhookEvent": "unknown", "issue": payload}

    issue = payload.get("issue") or {}
    fields = issue.get("fields") or {}
    user = payload.get("user") or {}

    return JiraEvent(
        webhook_event=payload.get("webhookEvent", "unknown"),
        issue_key=issue.get("key"),
        issue_id=str(issue.get("id")) if issue.get("id") is not None else None,
        summary=fields.get("summary"),
        status=(fields.get("status") or {}).get("name"),
        issue_type=(fields.get("issuetype") or {}).get("name"),
        project_key=(fields.get("project") or {}).get("key"),
        actor=user.get("displayName") or user.get("name"),
        payload=payload,
    )


def issue_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Simplify the embedded issue of a webhook payload, when present."""
    issue = payload.get("issue")
    if not isinstance(issue, dict):
        return None
    return _simplify_issue(issue)
