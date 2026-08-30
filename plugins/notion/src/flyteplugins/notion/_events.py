"""Normalized Notion change events.

Notion has no webhooks, so change events are produced by polling: the app
environment's `/api/poll` endpoint (or your own scheduled task) queries a
database for pages edited since a cursor and converts each result into a
`NotionEvent`. The model mirrors the webhook event models of the other
integration plugins so handlers and idempotent launching work the same way.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class NotionEvent(BaseModel):
    """A Notion change event produced by polling."""

    event_type: str = "page.edited"
    page_id: str
    database_id: str | None = None
    title: str = ""
    url: str | None = None
    last_edited_time: str | None = None
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = Field(default_factory=dict)

    @property
    def qualified_type(self) -> str:
        """The event type, e.g. `page.edited`."""
        return self.event_type

    def dedupe_key(self) -> str:
        """Stable key for idempotent run launching.

        Keyed on page + edit timestamp, so the same edit never launches twice
        while a later edit of the same page produces a new key.
        """
        base = f"{self.event_type}:{self.page_id}:{self.last_edited_time}"
        return hashlib.sha256(base.encode()).hexdigest()[:32]


def events_from_pages(
    pages: list[dict[str, Any]],
    *,
    database_id: str | None = None,
    event_type: str = "page.edited",
) -> list[NotionEvent]:
    """Convert simplified pages (from `query_database`) into `NotionEvent`s."""
    return [
        NotionEvent(
            event_type=event_type,
            page_id=page.get("id", ""),
            database_id=database_id or page.get("parent_id"),
            title=page.get("title", ""),
            url=page.get("url"),
            last_edited_time=page.get("last_edited_time"),
            payload=page,
        )
        for page in pages
    ]
