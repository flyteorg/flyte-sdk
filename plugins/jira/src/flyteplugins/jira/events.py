"""Jira webhook events, from the payload's `webhookEvent` field.

Some names take a `jira:` prefix and some do not — that inconsistency is Jira's.
These constants carry the exact wire values so you need not remember which.
"""

from __future__ import annotations

from flyte.extras.webhooks import EventType

__all__ = ["Comment", "Issue", "Project", "Sprint", "Version", "Worklog"]


class Issue(EventType):
    """Issue events. Note the `jira:` prefix, which comment events lack."""

    CREATED = "jira:issue_created"
    UPDATED = "jira:issue_updated"
    DELETED = "jira:issue_deleted"


class Comment(EventType):
    """Comment events. These carry no `jira:` prefix."""

    CREATED = "comment_created"
    UPDATED = "comment_updated"
    DELETED = "comment_deleted"


class Worklog(EventType):
    """Worklog events."""

    CREATED = "worklog_created"
    UPDATED = "worklog_updated"
    DELETED = "worklog_deleted"


class Project(EventType):
    """Project events."""

    CREATED = "project_created"
    UPDATED = "project_updated"
    DELETED = "project_deleted"


class Version(EventType):
    """Version (release) events."""

    CREATED = "jira:version_created"
    UPDATED = "jira:version_updated"
    RELEASED = "jira:version_released"
    UNRELEASED = "jira:version_unreleased"
    DELETED = "jira:version_deleted"


class Sprint(EventType):
    """Sprint events (Jira Software)."""

    CREATED = "sprint_created"
    UPDATED = "sprint_updated"
    STARTED = "sprint_started"
    CLOSED = "sprint_closed"
    DELETED = "sprint_deleted"
