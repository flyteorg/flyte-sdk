"""GitHub webhook signature verification and event normalization.

GitHub delivers webhooks with three headers this module cares about:

- `X-GitHub-Event` — the event type (`pull_request`, `issues`, `push`, ...).
- `X-GitHub-Delivery` — a UUID unique to this delivery, used as the dedupe key.
- `X-Hub-Signature-256` — `sha256=<hex>` HMAC of the raw body, computed with
  the webhook secret configured in the repository/org settings.

The receiver app verifies the signature against the secret mounted as an
environment variable, then normalizes the payload into a `GitHubEvent` that
event handlers can match on without knowing GitHub's schema details.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any, Mapping

from pydantic import BaseModel, Field

from ._errors import WebhookSignatureError

EVENT_HEADER = "X-GitHub-Event"
DELIVERY_HEADER = "X-GitHub-Delivery"
SIGNATURE_HEADER = "X-Hub-Signature-256"


class GitHubEvent(BaseModel):
    """A normalized GitHub webhook event."""

    event_type: str
    action: str | None = None
    delivery_id: str
    repository: str | None = None
    sender: str | None = None
    number: int | None = None
    title: str | None = None
    url: str | None = None
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = Field(default_factory=dict)

    @property
    def qualified_type(self) -> str:
        """`event_type.action` when an action is present, else `event_type`.

        Use this string to register handlers, e.g. `pull_request.opened` or
        `issues.closed`.
        """
        if self.action:
            return f"{self.event_type}.{self.action}"
        return self.event_type

    def dedupe_key(self) -> str:
        """Stable key for idempotent run launching.

        Keyed on event type + repository + issue/PR number when available (so
        retries of the same logical event dedupe), falling back to the unique
        delivery id for events without a number (e.g. `push`).
        """
        if self.number is not None and self.repository:
            base = f"{self.event_type}:{self.action or ''}:{self.repository}:{self.number}"
        else:
            base = f"{self.event_type}:{self.delivery_id}"
        return hashlib.sha256(base.encode()).hexdigest()[:32]


def verify_webhook_signature(payload: bytes, signature_header: str | None, secret: str) -> bool:
    """Verify the `X-Hub-Signature-256` header against the webhook secret.

    Args:
        payload: Raw request body bytes (exactly as received).
        signature_header: Value of the `X-Hub-Signature-256` header.
        secret: The webhook secret configured in GitHub.

    Returns:
        True when the signature matches; False otherwise (including missing or
        malformed headers).
    """
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header.removeprefix("sha256="))


def parse_webhook(headers: Mapping[str, str], body: bytes) -> GitHubEvent:
    """Parse webhook headers and body into a `GitHubEvent`.

    Header lookups are case-insensitive. Raises `WebhookSignatureError` when
    required headers are missing or the body is not valid JSON.
    """
    lowered = {k.lower(): v for k, v in headers.items()}
    event_type = lowered.get(EVENT_HEADER.lower())
    if not event_type:
        raise WebhookSignatureError(f"missing {EVENT_HEADER} header")
    delivery_id = lowered.get(DELIVERY_HEADER.lower(), "")

    try:
        payload = json.loads(body.decode("utf-8")) if body else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WebhookSignatureError(f"invalid webhook body: {exc}") from exc
    if not isinstance(payload, dict):
        payload = {"data": payload}

    repo = payload.get("repository") or {}
    issue_or_pr = payload.get("pull_request") or payload.get("issue") or {}
    sender = payload.get("sender") or {}
    comment = payload.get("comment") or {}

    title = issue_or_pr.get("title")
    url = comment.get("html_url") or issue_or_pr.get("html_url")
    number = issue_or_pr.get("number")
    if number is None:
        number = comment.get("id")

    return GitHubEvent(
        event_type=event_type,
        action=payload.get("action"),
        delivery_id=delivery_id,
        repository=repo.get("full_name"),
        sender=sender.get("login"),
        number=number,
        title=title,
        url=url,
        payload=payload,
    )
