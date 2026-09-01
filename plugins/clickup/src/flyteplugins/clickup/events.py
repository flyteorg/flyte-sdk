"""Typed constants for the ClickUp webhook events an app can subscribe to.

Register handlers with these instead of raw strings, so an editor can complete
them and a typo fails at import rather than by silently never matching:

```python
from flyteplugins.clickup import ClickUpAppEnvironment, events

app_env = ClickUpAppEnvironment(name="clickup-integration")

@app_env.on_event(events.Task.CREATED)
async def handle(event): ...
```

ClickUp event names are flat — there is no separate action field — so each
class simply groups the events for one kind of object.

These are `str` subclasses, so they are drop-in wherever a pattern string is
accepted. `on_event` still takes plain strings too — reach for one when ClickUp
ships an event these constants do not cover yet.
"""

from __future__ import annotations

import enum

__all__ = ["Folder", "Goal", "KeyResult", "List", "Space", "Task"]


class _EventType(str, enum.Enum):
    """Base for event constants: a real `str`, usable anywhere a pattern is."""

    # Without these, Python 3.11+ renders members as "Class.MEMBER" in
    # f-strings and str(), rather than the wire value handlers match on.
    __str__ = str.__str__
    __format__ = str.__format__  # type: ignore[assignment]


class Task(_EventType):
    """Task events."""

    CREATED = "taskCreated"
    UPDATED = "taskUpdated"
    DELETED = "taskDeleted"
    PRIORITY_UPDATED = "taskPriorityUpdated"
    STATUS_UPDATED = "taskStatusUpdated"
    ASSIGNEE_UPDATED = "taskAssigneeUpdated"
    DUE_DATE_UPDATED = "taskDueDateUpdated"
    TAG_UPDATED = "taskTagUpdated"
    MOVED = "taskMoved"
    COMMENT_POSTED = "taskCommentPosted"
    COMMENT_UPDATED = "taskCommentUpdated"
    TIME_ESTIMATE_UPDATED = "taskTimeEstimateUpdated"
    TIME_TRACKED_UPDATED = "taskTimeTrackedUpdated"


class List(_EventType):
    """List events."""

    CREATED = "listCreated"
    UPDATED = "listUpdated"
    DELETED = "listDeleted"


class Folder(_EventType):
    """Folder events."""

    CREATED = "folderCreated"
    UPDATED = "folderUpdated"
    DELETED = "folderDeleted"


class Space(_EventType):
    """Space events."""

    CREATED = "spaceCreated"
    UPDATED = "spaceUpdated"
    DELETED = "spaceDeleted"


class Goal(_EventType):
    """Goal events."""

    CREATED = "goalCreated"
    UPDATED = "goalUpdated"
    DELETED = "goalDeleted"


class KeyResult(_EventType):
    """Key-result (goal target) events."""

    CREATED = "keyResultCreated"
    UPDATED = "keyResultUpdated"
    DELETED = "keyResultDeleted"
