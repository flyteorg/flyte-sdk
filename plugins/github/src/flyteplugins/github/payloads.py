"""Typed views of GitHub payloads, for autocomplete inside handlers.

`event.payload` is `dict[str, Any]` — correct, since it carries GitHub's JSON
verbatim, but blind to write against. These TypedDicts spell the fields GitHub
actually sends, so `payloads.pull_request(event)` gives editors and agents
something to complete against:

```python
from flyteplugins.github import payloads

@app_env.on_event(events.PullRequest.OPENED)
async def on_primary(event):
    payload = payloads.pull_request(event)
    branch = payload["pull_request"]["head"]["ref"]
    repo = payload["repository"]["full_name"]
```

The helpers are casts, not validators: the dict is returned untouched, and a
field GitHub did not send is still a `KeyError` at runtime. Every class is
`total=False` because GitHub's payloads vary by action and App permissions —
treat presence the way you already would with a raw dict. Fields beyond these
still exist in the dict; the types name the commonly-read ones, not the whole
wire format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, cast

if TYPE_CHECKING:
    from flyte.extras.webhooks import WebhookEvent

__all__ = [
    "Comment",
    "GitRef",
    "Issue",
    "IssueCommentEvent",
    "Label",
    "PullRequest",
    "PullRequestEvent",
    "Repository",
    "User",
    "issue_comment",
    "pull_request",
]


class User(TypedDict, total=False):
    """An account — sender, author, assignee, owner."""

    login: str
    id: int
    type: str
    html_url: str


class Label(TypedDict, total=False):
    name: str
    color: str
    description: str


class Repository(TypedDict, total=False):
    full_name: str
    name: str
    html_url: str
    clone_url: str
    default_branch: str
    private: bool
    owner: User


class GitRef(TypedDict, total=False):
    """One side of a pull request — `head` or `base`."""

    ref: str
    sha: str
    label: str
    repo: Repository


class PullRequest(TypedDict, total=False):
    """`payload["pull_request"]` — the PR the event is about."""

    number: int
    title: str
    body: str
    state: str
    draft: bool
    merged: bool
    html_url: str
    user: User
    labels: list[Label]
    head: GitRef
    base: GitRef
    additions: int
    deletions: int
    changed_files: int


class Issue(TypedDict, total=False):
    """`payload["issue"]` — present on `issues` and `issue_comment` events.

    A comment on a pull request also arrives as `issue_comment`; the issue then
    carries a `pull_request` key, which is how the two are told apart.
    """

    number: int
    title: str
    body: str
    state: str
    html_url: str
    user: User
    labels: list[Label]
    pull_request: dict[str, Any]


class Comment(TypedDict, total=False):
    id: int
    body: str
    html_url: str
    user: User


class PullRequestEvent(TypedDict, total=False):
    """A `pull_request` delivery."""

    action: str
    number: int
    pull_request: PullRequest
    repository: Repository
    sender: User
    installation: dict[str, Any]


class IssueCommentEvent(TypedDict, total=False):
    """An `issue_comment` delivery — the comment-triggered-agent shape."""

    action: str
    issue: Issue
    comment: Comment
    repository: Repository
    sender: User
    installation: dict[str, Any]


def pull_request(event: WebhookEvent) -> PullRequestEvent:
    """Typed view of a `pull_request` delivery's payload. A cast, not validation."""
    return cast("PullRequestEvent", event.payload)


def issue_comment(event: WebhookEvent) -> IssueCommentEvent:
    """Typed view of an `issue_comment` delivery's payload. A cast, not validation."""
    return cast("IssueCommentEvent", event.payload)
