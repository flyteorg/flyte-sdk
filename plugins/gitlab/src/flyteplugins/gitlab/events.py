"""GitLab webhook events, from the payload's `object_kind` plus `object_attributes.action`.

GitLab names each hook after its object; the header is `X-Gitlab-Event`
("Merge Request Hook"), and the payload's `object_kind` carries the same idea in
snake_case (`merge_request`). The action then lives on
`object_attributes.action` ("open", "close", ...). These constants spell the
wire values so a handler can match without assembling either part by hand.

Note hooks (comments) have no action of their own, and push / tag-push hooks
send none either — those classes expose only `ANY`. Pipelines and deployments
report a `status` instead of an action, and deployment / release hooks put it
at the top level of the payload rather than under `object_attributes`; the
parser folds all of that into the same `kind.action` shape used here.
"""

from __future__ import annotations

from flyte.extras.webhooks import EventType

__all__ = [
    "Deployment",
    "Issue",
    "MergeRequest",
    "Note",
    "Pipeline",
    "Push",
    "Release",
    "TagPush",
]


class MergeRequest(EventType):
    """`merge_request` events. GitLab marks merge requests with `!` (`octo/repo!7`); issues use `#`."""

    ANY = "merge_request"
    OPEN = "merge_request.open"
    CLOSE = "merge_request.close"
    REOPEN = "merge_request.reopen"
    UPDATE = "merge_request.update"
    APPROVAL = "merge_request.approval"
    APPROVED = "merge_request.approved"
    UNAPPROVAL = "merge_request.unapproval"
    UNAPPROVED = "merge_request.unapproved"
    MERGE = "merge_request.merge"


class Issue(EventType):
    """`issue` events."""

    ANY = "issue"
    OPEN = "issue.open"
    CLOSE = "issue.close"
    REOPEN = "issue.reopen"
    UPDATE = "issue.update"


class Note(EventType):
    """`note` events — comments on issues, merge requests, commits, or snippets.

    GitLab sends no action for notes. `object_attributes.noteable_type` says what
    the comment is on; the note's own `id` keeps two comments on one thread
    from collapsing onto a single dedupe key.
    """

    ANY = "note"


class Push(EventType):
    """`push` events — a branch push to a project. No action."""

    ANY = "push"


class TagPush(EventType):
    """`tag_push` events — a tag created or deleted. No action."""

    ANY = "tag_push"


class Pipeline(EventType):
    """`pipeline` events. The action is the pipeline's `status`."""

    ANY = "pipeline"
    CREATED = "pipeline.created"
    WAITING_FOR_RESOURCE = "pipeline.waiting_for_resource"
    PREPARING = "pipeline.preparing"
    PENDING = "pipeline.pending"
    RUNNING = "pipeline.running"
    SUCCESS = "pipeline.success"
    FAILED = "pipeline.failed"
    CANCELED = "pipeline.canceled"
    SKIPPED = "pipeline.skipped"
    MANUAL = "pipeline.manual"
    SCHEDULED = "pipeline.scheduled"


class Release(EventType):
    """`release` events. The action sits at the top level of the payload."""

    ANY = "release"
    CREATE = "release.create"
    UPDATE = "release.update"


class Deployment(EventType):
    """`deployment` events. The action is the top-level `status`."""

    ANY = "deployment"
    RUNNING = "deployment.running"
    SUCCESS = "deployment.success"
    FAILED = "deployment.failed"
    CANCELED = "deployment.canceled"
