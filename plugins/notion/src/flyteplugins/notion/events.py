"""Typed constants for the Notion change events an app can subscribe to.

Register handlers with these instead of raw strings, so an editor can complete
them and a typo fails at import rather than by silently never matching:

```python
from flyteplugins.notion import NotionAppEnvironment, events

app_env = NotionAppEnvironment(name="notion-integration")

@app_env.on_event(events.Page.EDITED)
async def handle(event): ...
```

Notion has no webhooks, so change events come from polling: the app's poll
endpoint compares each page's `last_edited_time` against a cursor. That yields
a single event type today, but the constant keeps handler registration
consistent with the other integration plugins and leaves room to grow.

These are `str` subclasses, so they are drop-in wherever a pattern string is
accepted. `on_event` still takes plain strings too — reach for one when Notion
ships an event these constants do not cover yet.
"""

from __future__ import annotations

import enum

__all__ = ["Page"]


class _EventType(str, enum.Enum):
    """Base for event constants: a real `str`, usable anywhere a pattern is."""

    # Without these, Python 3.11+ renders members as "Class.MEMBER" in
    # f-strings and str(), rather than the wire value handlers match on.
    __str__ = str.__str__
    __format__ = str.__format__  # type: ignore[assignment]


class Page(_EventType):
    """Page change events produced by polling a database."""

    EDITED = "page.edited"
    """A page's `last_edited_time` advanced past the poll cursor."""
