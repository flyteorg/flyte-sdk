"""Linear webhook signature verification and event normalization.

Linear delivers webhooks with an `X-Linear-Signature` header containing the
hex HMAC-SHA256 of the raw body, computed with the signing secret shown when
the webhook is created. Payloads carry an `action` (`create`/`update`/
`remove`), a `type` (`Issue`, `Comment`, ...), and the entity under `data`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any, Mapping

from pydantic import BaseModel, Field

from ._errors import WebhookSignatureError

SIGNATURE_HEADER = "X-Linear-Signature"


class LinearEvent(BaseModel):
    """A normalized Linear webhook event."""

    action: str
    entity_type: str
    entity_id: str | None = None
    updated_at: str | None = None
    title: str | None = None
    entity_url: str | None = None
    team_id: str | None = None
    state_id: str | None = None
    organization: str | None = None
    webhook_id: str | None = None
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = Field(default_factory=dict)

    @property
    def qualified_type(self) -> str:
        """`Type.action`, e.g. `Issue.create` or `Comment.update`."""
        return f"{self.entity_type}.{self.action}"

    def dedupe_key(self) -> str:
        """Stable key for idempotent run launching.

        Keyed on entity + action + the entity's update timestamp, so a
        redelivery of one webhook dedupes while a *later* change to the same
        entity gets its own key. The timestamp is what makes this usable for
        `update` actions: keyed on the entity alone, every subsequent update to
        an issue would collapse onto the first one's key and never launch.
        """
        base = f"{self.entity_type}:{self.action}:{self.entity_id or self.webhook_id}:{self.updated_at or ''}"
        return hashlib.sha256(base.encode()).hexdigest()[:32]


def verify_webhook_signature(payload: bytes, signature_header: str | None, secret: str) -> bool:
    """Verify the `X-Linear-Signature` header against the webhook secret."""
    if not signature_header:
        return False
    expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    # Compare as bytes: compare_digest rejects str operands containing non-ASCII, and the
    # header is attacker-controlled, so a str comparison would raise instead of returning False.
    return hmac.compare_digest(expected.encode("utf-8"), signature_header.strip().encode("utf-8"))


def parse_webhook(headers: Mapping[str, str], body: bytes) -> LinearEvent:
    """Parse webhook headers and body into a `LinearEvent`.

    Header lookups are case-insensitive. Raises `WebhookSignatureError` when
    the body is not valid JSON.
    """
    try:
        payload = json.loads(body.decode("utf-8")) if body else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WebhookSignatureError(f"invalid webhook body: {exc}") from exc
    if not isinstance(payload, dict):
        payload = {"data": payload}

    data = payload.get("data") or {}
    organization = payload.get("organization") or {}

    return LinearEvent(
        action=payload.get("action", "unknown"),
        entity_type=payload.get("type", "Unknown"),
        entity_id=data.get("id"),
        # `updatedAt` is on the entity; `createdAt` is the webhook delivery time and
        # is the only timestamp on payloads whose entity does not carry one.
        updated_at=data.get("updatedAt") or payload.get("createdAt"),
        title=data.get("title"),
        entity_url=data.get("url") or payload.get("url"),
        team_id=_team_id(data),
        state_id=data.get("stateId"),
        organization=organization.get("name"),
        webhook_id=payload.get("webhookId"),
        payload=payload,
    )


def _team_id(data: dict[str, Any]) -> str | None:
    """Find the team id in an entity payload.

    Issue payloads carry `teamId` directly; Comment and Reaction payloads carry
    it only on the nested issue. Without these fallbacks a `team_ids` allowlist
    would drop every non-Issue event as unattributable.
    """
    issue = data.get("issue") or {}
    for candidate in (
        data.get("teamId"),
        (data.get("team") or {}).get("id"),
        issue.get("teamId"),
        (issue.get("team") or {}).get("id"),
    ):
        if candidate:
            return str(candidate)
    return None
