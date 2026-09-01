"""Typed constants for the GitHub webhook events an app can subscribe to.

Register handlers with these instead of raw strings, so an editor can complete
them and a typo fails at import rather than by silently never matching:

```python
from flyteplugins.github import GitHubAppEnvironment, events

app_env = GitHubAppEnvironment(name="github-integration")

@app_env.on_event(events.PullRequest.OPENED)
async def triage(event): ...
```

Each class is one GitHub event type (the `X-GitHub-Event` header) and its
members are that type's `action` values, spelled as the `type.action` pattern
`on_event` matches on. `ANY` is the bare event type, matching every action:

```python
@app_env.on_event(events.PullRequest.ANY)      # every pull_request action
@app_env.on_event(events.PullRequest.OPENED)   # just pull_request.opened
```

These are `str` subclasses, so they are drop-in wherever a pattern string is
accepted. `on_event` still takes plain strings too — reach for one when GitHub
ships an event these constants do not cover yet.
"""

from __future__ import annotations

import enum

__all__ = [
    "CheckRun",
    "CheckSuite",
    "Create",
    "Delete",
    "Fork",
    "IssueComment",
    "Issues",
    "PullRequest",
    "PullRequestReview",
    "PullRequestReviewComment",
    "Push",
    "Release",
    "Star",
    "WorkflowRun",
]


class _EventType(str, enum.Enum):
    """Base for event constants: a real `str`, usable anywhere a pattern is."""

    # Without these, Python 3.11+ renders members as "PullRequest.OPENED" in
    # f-strings and str(), rather than the wire value handlers match on.
    __str__ = str.__str__
    __format__ = str.__format__  # type: ignore[assignment]


class PullRequest(_EventType):
    """`pull_request` events."""

    ANY = "pull_request"
    OPENED = "pull_request.opened"
    CLOSED = "pull_request.closed"
    """Fires on both merge and close-without-merge; check `payload["pull_request"]["merged"]`."""
    REOPENED = "pull_request.reopened"
    EDITED = "pull_request.edited"
    ASSIGNED = "pull_request.assigned"
    UNASSIGNED = "pull_request.unassigned"
    LABELED = "pull_request.labeled"
    UNLABELED = "pull_request.unlabeled"
    SYNCHRONIZE = "pull_request.synchronize"
    """New commits were pushed to the PR's head branch."""
    READY_FOR_REVIEW = "pull_request.ready_for_review"
    CONVERTED_TO_DRAFT = "pull_request.converted_to_draft"
    REVIEW_REQUESTED = "pull_request.review_requested"
    REVIEW_REQUEST_REMOVED = "pull_request.review_request_removed"
    LOCKED = "pull_request.locked"
    UNLOCKED = "pull_request.unlocked"


class Issues(_EventType):
    """`issues` events. Note GitHub's event type is plural."""

    ANY = "issues"
    OPENED = "issues.opened"
    CLOSED = "issues.closed"
    REOPENED = "issues.reopened"
    EDITED = "issues.edited"
    ASSIGNED = "issues.assigned"
    UNASSIGNED = "issues.unassigned"
    LABELED = "issues.labeled"
    UNLABELED = "issues.unlabeled"
    MILESTONED = "issues.milestoned"
    DEMILESTONED = "issues.demilestoned"
    PINNED = "issues.pinned"
    UNPINNED = "issues.unpinned"
    LOCKED = "issues.locked"
    UNLOCKED = "issues.unlocked"
    TRANSFERRED = "issues.transferred"
    DELETED = "issues.deleted"


class IssueComment(_EventType):
    """`issue_comment` events, on both issues and pull requests."""

    ANY = "issue_comment"
    CREATED = "issue_comment.created"
    EDITED = "issue_comment.edited"
    DELETED = "issue_comment.deleted"


class PullRequestReview(_EventType):
    """`pull_request_review` events."""

    ANY = "pull_request_review"
    SUBMITTED = "pull_request_review.submitted"
    EDITED = "pull_request_review.edited"
    DISMISSED = "pull_request_review.dismissed"


class PullRequestReviewComment(_EventType):
    """`pull_request_review_comment` events — inline comments on a diff."""

    ANY = "pull_request_review_comment"
    CREATED = "pull_request_review_comment.created"
    EDITED = "pull_request_review_comment.edited"
    DELETED = "pull_request_review_comment.deleted"


class Push(_EventType):
    """`push` events. GitHub sends no `action`, so there is only `ANY`."""

    ANY = "push"


class Create(_EventType):
    """`create` events — a branch or tag was created. No `action`."""

    ANY = "create"


class Delete(_EventType):
    """`delete` events — a branch or tag was deleted. No `action`."""

    ANY = "delete"


class Fork(_EventType):
    """`fork` events. No `action`."""

    ANY = "fork"


class Release(_EventType):
    """`release` events."""

    ANY = "release"
    PUBLISHED = "release.published"
    UNPUBLISHED = "release.unpublished"
    CREATED = "release.created"
    EDITED = "release.edited"
    DELETED = "release.deleted"
    PRERELEASED = "release.prereleased"
    RELEASED = "release.released"


class WorkflowRun(_EventType):
    """`workflow_run` events — GitHub Actions run lifecycle."""

    ANY = "workflow_run"
    REQUESTED = "workflow_run.requested"
    IN_PROGRESS = "workflow_run.in_progress"
    COMPLETED = "workflow_run.completed"


class CheckRun(_EventType):
    """`check_run` events."""

    ANY = "check_run"
    CREATED = "check_run.created"
    COMPLETED = "check_run.completed"
    REREQUESTED = "check_run.rerequested"
    REQUESTED_ACTION = "check_run.requested_action"


class CheckSuite(_EventType):
    """`check_suite` events."""

    ANY = "check_suite"
    COMPLETED = "check_suite.completed"
    REQUESTED = "check_suite.requested"
    REREQUESTED = "check_suite.rerequested"


class Star(_EventType):
    """`star` events."""

    ANY = "star"
    CREATED = "star.created"
    DELETED = "star.deleted"
