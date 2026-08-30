"""ClickUp webhook signature verification and event normalization.

ClickUp delivers webhooks with an `x-clickup-signature` header containing the
hex HMAC-SHA256 of the raw body, computed with the signing secret shown when
the webhook is created. Payloads carry an `event` name (`taskCreated`,
`taskStatusUpdated`, `taskCommented`, ...) plus the affected task.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any, Mapping

from pydantic import BaseModel, Field

from ._errors import WebhookSignatureError

SIGNATURE_HEADER = "x-clickup-signature"


class ClickUpEvent(BaseModel):
    """A normalized ClickUp webhook event."""

    event: str
    task_id: str | None = None
    list_id: str | None = None
    task_name: str | None = None
    task_status: str | None = None
    task_url: str | None = None
    webhook_id: str | None = None
    event_timestamp: int | None = None
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = Field(default_factory=dict)

    @property
    def qualified_type(self) -> str:
        """The ClickUp event name, e.g. `taskStatusUpdated`."""
        return self.event

    def dedupe_key(self) -> str:
        """Stable key for idempotent run launching.

        Keyed on event + task + ClickUp's own event timestamp, so retries of
        the same delivery dedupe while later updates to the same task produce
        distinct keys.
        """
        base = f"{self.event}:{self.task_id}:{self.event_timestamp or self.webhook_id}"
        return hashlib.sha256(base.encode()).hexdigest()[:32]


def verify_webhook_signature(payload: bytes, signature_header: str | None, secret: str) -> bool:
    """Verify the `x-clickup-signature` header against the webhook secret."""
    if not signature_header:
        return False
    expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header.strip())


def parse_webhook(headers: Mapping[str, str], body: bytes) -> ClickUpEvent:
    """Parse webhook headers and body into a `ClickUpEvent`.

    Raises `WebhookSignatureError` when the body is not valid JSON.
    """
    try:
        payload = json.loads(body.decode("utf-8")) if body else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WebhookSignatureError(f"invalid webhook body: {exc}") from exc
    if not isinstance(payload, dict):
        payload = {"event": "unknown", "data": payload}

    task = payload.get("task") or {}
    status = (task.get("status") or {}).get("status")

    return ClickUpEvent(
        event=payload.get("event", "unknown"),
        task_id=str(payload.get("task_id")) if payload.get("task_id") is not None else task.get("id"),
        list_id=str(payload.get("list_id")) if payload.get("list_id") is not None else None,
        task_name=task.get("name"),
        task_status=status,
        task_url=task.get("url"),
        webhook_id=payload.get("webhook_id"),
        event_timestamp=payload.get("timestamp"),
        payload=payload,
    )
