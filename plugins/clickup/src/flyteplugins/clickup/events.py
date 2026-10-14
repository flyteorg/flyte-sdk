"""ClickUp webhook events. Names are flat — ClickUp sends no separate action."""

from __future__ import annotations

from flyte.extras.webhooks import EventType

__all__ = ["Folder", "Goal", "KeyResult", "List", "Space", "Task"]


class Task(EventType):
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


class List(EventType):
    """List events."""

    CREATED = "listCreated"
    UPDATED = "listUpdated"
    DELETED = "listDeleted"


class Folder(EventType):
    """Folder events."""

    CREATED = "folderCreated"
    UPDATED = "folderUpdated"
    DELETED = "folderDeleted"


class Space(EventType):
    """Space events."""

    CREATED = "spaceCreated"
    UPDATED = "spaceUpdated"
    DELETED = "spaceDeleted"


class Goal(EventType):
    """Goal events."""

    CREATED = "goalCreated"
    UPDATED = "goalUpdated"
    DELETED = "goalDeleted"


class KeyResult(EventType):
    """Key-result (goal target) events."""

    CREATED = "keyResultCreated"
    UPDATED = "keyResultUpdated"
    DELETED = "keyResultDeleted"
