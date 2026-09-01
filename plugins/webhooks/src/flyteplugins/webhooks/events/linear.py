"""Linear webhook events, spelled as the `Type.action` pattern Linear sends."""

from __future__ import annotations

from ._base import EventType

__all__ = ["Attachment", "Comment", "Cycle", "Issue", "IssueLabel", "Project", "ProjectUpdate", "Reaction"]


class Issue(EventType):
    """`Issue` entity events."""

    ANY = "Issue"
    CREATE = "Issue.create"
    UPDATE = "Issue.update"
    REMOVE = "Issue.remove"


class Comment(EventType):
    """`Comment` entity events."""

    ANY = "Comment"
    CREATE = "Comment.create"
    UPDATE = "Comment.update"
    REMOVE = "Comment.remove"


class IssueLabel(EventType):
    """`IssueLabel` entity events."""

    ANY = "IssueLabel"
    CREATE = "IssueLabel.create"
    UPDATE = "IssueLabel.update"
    REMOVE = "IssueLabel.remove"


class Project(EventType):
    """`Project` entity events."""

    ANY = "Project"
    CREATE = "Project.create"
    UPDATE = "Project.update"
    REMOVE = "Project.remove"


class ProjectUpdate(EventType):
    """`ProjectUpdate` entity events — project status posts."""

    ANY = "ProjectUpdate"
    CREATE = "ProjectUpdate.create"
    UPDATE = "ProjectUpdate.update"
    REMOVE = "ProjectUpdate.remove"


class Cycle(EventType):
    """`Cycle` entity events."""

    ANY = "Cycle"
    CREATE = "Cycle.create"
    UPDATE = "Cycle.update"
    REMOVE = "Cycle.remove"


class Reaction(EventType):
    """`Reaction` entity events."""

    ANY = "Reaction"
    CREATE = "Reaction.create"
    UPDATE = "Reaction.update"
    REMOVE = "Reaction.remove"


class Attachment(EventType):
    """`Attachment` entity events."""

    ANY = "Attachment"
    CREATE = "Attachment.create"
    UPDATE = "Attachment.update"
    REMOVE = "Attachment.remove"
