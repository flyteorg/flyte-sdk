"""Typed constants for the Jira webhook events an app can subscribe to.

Register handlers with these instead of raw strings, so an editor can complete
them and a typo fails at import rather than by silently never matching:

```python
from flyteplugins.jira import JiraAppEnvironment, events

app_env = JiraAppEnvironment(name="jira-integration")

@app_env.on_event(events.Issue.CREATED)
async def handle(event): ...
```

Jira sends the event name in the payload's `webhookEvent` field. Some are
prefixed `jira:` and some are not — that inconsistency is Jira's, and these
constants carry the exact wire values so you do not have to remember which is
which.

These are `str` subclasses, so they are drop-in wherever a pattern string is
accepted. `on_event` still takes plain strings too — reach for one when Jira
ships an event these constants do not cover yet.
"""

from __future__ import annotations

import enum

__all__ = ["Comment", "Issue", "Project", "Sprint", "Version", "Worklog"]


class _EventType(str, enum.Enum):
    """Base for event constants: a real `str`, usable anywhere a pattern is."""

    # Without these, Python 3.11+ renders members as "Class.MEMBER" in
    # f-strings and str(), rather than the wire value handlers match on.
    __str__ = str.__str__
    __format__ = str.__format__  # type: ignore[assignment]


class Issue(_EventType):
    """Issue events. Note the `jira:` prefix, which comment events lack."""

    CREATED = "jira:issue_created"
    UPDATED = "jira:issue_updated"
    DELETED = "jira:issue_deleted"


class Comment(_EventType):
    """Comment events. These carry no `jira:` prefix."""

    CREATED = "comment_created"
    UPDATED = "comment_updated"
    DELETED = "comment_deleted"


class Worklog(_EventType):
    """Worklog events."""

    CREATED = "worklog_created"
    UPDATED = "worklog_updated"
    DELETED = "worklog_deleted"


class Project(_EventType):
    """Project events."""

    CREATED = "project_created"
    UPDATED = "project_updated"
    DELETED = "project_deleted"


class Version(_EventType):
    """Version (release) events."""

    CREATED = "jira:version_created"
    UPDATED = "jira:version_updated"
    RELEASED = "jira:version_released"
    UNRELEASED = "jira:version_unreleased"
    DELETED = "jira:version_deleted"


class Sprint(_EventType):
    """Sprint events (Jira Software)."""

    CREATED = "sprint_created"
    UPDATED = "sprint_updated"
    STARTED = "sprint_started"
    CLOSED = "sprint_closed"
    DELETED = "sprint_deleted"
