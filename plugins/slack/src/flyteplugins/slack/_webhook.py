"""Slack Events API signature verification and event normalization.

Slack delivers Events API requests with two headers this module cares about:

- `X-Slack-Request-Timestamp` — Unix seconds when the request was sent.
- `X-Slack-Signature` — `v0=<hex>` HMAC of `v0:<timestamp>:<raw body>`,
  computed with the app's signing secret.

Requests older than a few minutes are rejected as replay protection. Before
the Events API is wired up, Slack sends a one-off `url_verification` payload
that the receiver must echo back as the `challenge`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from typing import Any, Mapping

from pydantic import BaseModel, Field

from ._errors import EventSignatureError

SIGNATURE_HEADER = "X-Slack-Signature"
TIMESTAMP_HEADER = "X-Slack-Request-Timestamp"

#: Reject requests whose timestamp is older than this many seconds.
MAX_REQUEST_AGE_SECONDS = 60 * 5


class SlackEvent(BaseModel):
    """A normalized Slack Events API event."""

    event_type: str
    subtype: str | None = None
    channel: str | None = None
    ts: str | None = None
    thread_ts: str | None = None
    user: str | None = None
    text: str = ""
    reaction: str | None = None
    team_id: str | None = None
    event_id: str = ""
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = Field(default_factory=dict)

    @property
    def qualified_type(self) -> str:
        """`event_type.subtype` when a subtype is present, else `event_type`."""
        if self.subtype:
            return f"{self.event_type}.{self.subtype}"
        return self.event_type

    @property
    def root_ts(self) -> str | None:
        """Timestamp of the thread root: `thread_ts` when set, else `ts`."""
        return self.thread_ts or self.ts

    def dedupe_key(self) -> str:
        """Stable key for idempotent run launching.

        Threaded events collapse to their root so every message in one thread
        maps to the same key; events without a channel/ts fall back to the
        Slack event id.
        """
        if self.channel and self.root_ts:
            base = f"{self.event_type}:{self.channel}:{self.root_ts}"
        else:
            base = f"{self.event_type}:{self.event_id}"
        return hashlib.sha256(base.encode()).hexdigest()[:32]


def verify_event_signature(
    body: bytes,
    timestamp_header: str | None,
    signature_header: str | None,
    secret: str,
    *,
    max_age_seconds: int = MAX_REQUEST_AGE_SECONDS,
    now: float | None = None,
) -> bool:
    """Verify the `X-Slack-Signature` header against the signing secret.

    Args:
        body: Raw request body bytes.
        timestamp_header: Value of `X-Slack-Request-Timestamp`.
        signature_header: Value of `X-Slack-Signature` (`v0=<hex>`).
        secret: The Slack app signing secret.
        max_age_seconds: Reject timestamps older than this (replay protection).
        now: Injectable clock for tests (Unix seconds).

    Returns:
        True when the signature matches and the timestamp is fresh.
    """
    if not timestamp_header or not signature_header or not signature_header.startswith("v0="):
        return False
    try:
        timestamp = int(timestamp_header)
    except ValueError:
        return False
    if abs((now if now is not None else time.time()) - timestamp) > max_age_seconds:
        return False
    basestring = f"v0:{timestamp}:{body.decode('utf-8', errors='replace')}".encode()
    expected = "v0=" + hmac.new(secret.encode(), basestring, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)


def parse_url_verification(body: bytes) -> str | None:
    """Return the `challenge` if this is a URL verification request, else None."""
    try:
        data = json.loads(body.decode("utf-8")) if body else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(data, dict) and data.get("type") == "url_verification":
        return str(data.get("challenge", ""))
    return None


def parse_event(headers: Mapping[str, str], body: bytes) -> SlackEvent:
    """Parse an event-callback payload into a `SlackEvent`.

    Header lookups are case-insensitive. Raises `EventSignatureError` when the
    body is not a valid event callback.
    """
    try:
        data = json.loads(body.decode("utf-8")) if body else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EventSignatureError(f"invalid event body: {exc}") from exc
    if not isinstance(data, dict):
        data = {"event": data}

    event = data.get("event") or {}
    if not isinstance(event, dict) or not event:
        raise EventSignatureError("event payload is missing its `event` object")

    return SlackEvent(
        event_type=event.get("type", "unknown"),
        subtype=event.get("subtype"),
        channel=event.get("channel"),
        ts=event.get("ts"),
        thread_ts=event.get("thread_ts"),
        user=event.get("user"),
        text=event.get("text") or "",
        reaction=event.get("reaction"),
        team_id=data.get("team_id"),
        event_id=data.get("event_id", ""),
        payload=data,
    )
