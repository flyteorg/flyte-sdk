"""The normalized webhook event.

Five SaaS products, five payload shapes, one model. Handlers match on
`qualified_type` and read the fields they care about; `payload` always carries
the provider's original JSON for anything this model does not surface.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, computed_field


class WebhookEvent(BaseModel):
    """One inbound webhook, normalized across providers.

    Args:
        provider: Which integration delivered this (`github`, `slack`, ...).
        event_type: The provider's event type (`pull_request`, `Issue`, ...).
        action: The provider's action within that type (`opened`, `create`, ...),
            when it splits the two. None for providers that do not.
        delivery_id: The provider's own id for this delivery, where it sends one.
        resource_id: The thing the event is about — issue key, task id, message
            timestamp. Used for dedupe and shown on the dashboard.
        occurred_at: The provider's timestamp for the change, when it sends one.
            Folded into the dedupe key so a *later* change to the same resource
            gets its own key.
        scope: The container the resource lives in — repository, channel, team,
            list, project. Matched against the app's allowlist.
        title: Human-readable summary, for the dashboard.
        url: Link back to the resource in the provider's UI.
        actor: Who caused the event.
        payload: The provider's original JSON, verbatim.
    """

    provider: str
    event_type: str
    action: str | None = None
    delivery_id: str = ""
    resource_id: str | None = None
    occurred_at: str | None = None
    scope: str | None = None
    title: str | None = None
    url: str | None = None
    actor: str | None = None
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def qualified_type(self) -> str:
        """`type.action` when the provider splits the two, else `type`.

        This is what handlers register against, and what the `events` constants
        spell out. A computed field rather than a plain property, so it appears
        in `/api/events` — a consumer of that endpoint should not have to
        reassemble it from `event_type` and `action`.
        """
        if self.action:
            return f"{self.event_type}.{self.action}"
        return self.event_type

    def dedupe_key(self) -> str:
        """A stable key for `flyte.extras.webhooks.idempotent_run`.

        Keyed on provider + qualified type + resource + the provider's own
        timestamp. The timestamp is what makes this usable for `update`-shaped
        events: without it, every later change to one resource would collapse
        onto the first one's key and never launch. Events with no resource fall
        back to the delivery id, which is unique per delivery.

        The key is just a string — build your own and pass it to
        `idempotent_run` directly when you want a different scope, such as one
        run per thread rather than one per message.
        """
        if self.resource_id:
            base = f"{self.provider}:{self.qualified_type}:{self.resource_id}:{self.occurred_at or ''}"
        else:
            base = f"{self.provider}:{self.qualified_type}:{self.delivery_id}"
        return hashlib.sha256(base.encode()).hexdigest()[:32]
