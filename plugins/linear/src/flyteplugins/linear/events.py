"""Typed constants for the Linear webhook events an app can subscribe to.

Register handlers with these instead of raw strings, so an editor can complete
them and a typo fails at import rather than by silently never matching:

```python
from flyteplugins.linear import LinearAppEnvironment, events

app_env = LinearAppEnvironment(name="linear-integration")

@app_env.on_event(events.Issue.CREATE)
async def handle(event): ...
```

Each class is one Linear entity type and its members are the `create` /
`update` / `remove` actions Linear sends for it, spelled as the `Type.action`
pattern `on_event` matches on. `ANY` is the bare entity type, matching every
action on it.

These are `str` subclasses, so they are drop-in wherever a pattern string is
accepted. `on_event` still takes plain strings too — reach for one when Linear
ships an event these constants do not cover yet.
"""

from __future__ import annotations

import enum

__all__ = [
    "Attachment",
    "Comment",
    "Cycle",
    "Issue",
    "IssueLabel",
    "Project",
    "ProjectUpdate",
    "Reaction",
]


class _EventType(str, enum.Enum):
    """Base for event constants: a real `str`, usable anywhere a pattern is."""

    # Without these, Python 3.11+ renders members as "Class.MEMBER" in
    # f-strings and str(), rather than the wire value handlers match on.
    __str__ = str.__str__
    __format__ = str.__format__  # type: ignore[assignment]


class Issue(_EventType):
    """`Issue` entity events."""

    ANY = "Issue"
    CREATE = "Issue.create"
    UPDATE = "Issue.update"
    REMOVE = "Issue.remove"


class Comment(_EventType):
    """`Comment` entity events."""

    ANY = "Comment"
    CREATE = "Comment.create"
    UPDATE = "Comment.update"
    REMOVE = "Comment.remove"


class IssueLabel(_EventType):
    """`IssueLabel` entity events."""

    ANY = "IssueLabel"
    CREATE = "IssueLabel.create"
    UPDATE = "IssueLabel.update"
    REMOVE = "IssueLabel.remove"


class Project(_EventType):
    """`Project` entity events."""

    ANY = "Project"
    CREATE = "Project.create"
    UPDATE = "Project.update"
    REMOVE = "Project.remove"


class ProjectUpdate(_EventType):
    """`ProjectUpdate` entity events — project status posts."""

    ANY = "ProjectUpdate"
    CREATE = "ProjectUpdate.create"
    UPDATE = "ProjectUpdate.update"
    REMOVE = "ProjectUpdate.remove"


class Cycle(_EventType):
    """`Cycle` entity events."""

    ANY = "Cycle"
    CREATE = "Cycle.create"
    UPDATE = "Cycle.update"
    REMOVE = "Cycle.remove"


class Reaction(_EventType):
    """`Reaction` entity events."""

    ANY = "Reaction"
    CREATE = "Reaction.create"
    UPDATE = "Reaction.update"
    REMOVE = "Reaction.remove"


class Attachment(_EventType):
    """`Attachment` entity events."""

    ANY = "Attachment"
    CREATE = "Attachment.create"
    UPDATE = "Attachment.update"
    REMOVE = "Attachment.remove"
