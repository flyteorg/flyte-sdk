"""Typed views of GitLab payloads, for autocomplete inside handlers.

`event.payload` is `dict[str, Any]` — correct, since it carries GitLab's JSON
verbatim, but blind to write against. These TypedDicts spell the fields GitLab
actually sends, so `payloads.merge_request(event)` gives editors and agents
something to complete against:

```python
from flyteplugins.gitlab import payloads

@app_env.on_event(events.MergeRequest.OPEN)
async def on_mr(event):
    payload = payloads.merge_request(event)
    author = payload["user"]["username"]
    branch = payload["object_attributes"]["source_branch"]
```

The helpers are casts, not validators: the dict is returned untouched, and a
field GitLab did not send is still a `KeyError` at runtime. Every class is
`total=False` because GitLab's payloads vary by hook and trigger — treat
presence the way you already would with a raw dict. Fields beyond these still
exist in the dict; the types name the commonly-read ones, not the whole wire
format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, cast

if TYPE_CHECKING:
    from flyte.extras.webhooks import WebhookEvent

__all__ = [
    "Author",
    "MergeRequest",
    "MergeRequestEvent",
    "NoteEvent",
    "ObjectAttributes",
    "Project",
    "Repository",
    "User",
    "merge_request",
    "note",
]


class User(TypedDict, total=False):
    """The actor on issue / merge request / note hooks."""

    name: str
    username: str
    id: int
    avatar_url: str


class Author(TypedDict, total=False):
    """A commit or push author (a plain string on some hooks)."""

    name: str
    email: str


class Project(TypedDict, total=False):
    """`payload["project"]` — the repo the event is about."""

    id: int
    name: str
    path_with_namespace: str
    default_branch: str
    web_url: str
    visibility: str


class Repository(TypedDict, total=False):
    """`payload["repository"]` — the repo as push hooks call it."""

    name: str
    url: str
    path: str
    homepage: str


class ObjectAttributes(TypedDict, total=False):
    """`payload["object_attributes"]` — the object that changed."""

    id: int
    iid: int
    action: str
    state: str
    title: str
    description: str
    url: str
    source_branch: str
    target_branch: str
    merge_status: str
    updated_at: str
    created_at: str


class MergeRequestEvent(TypedDict, total=False):
    """A `merge_request` delivery."""

    object_kind: str
    event_type: str
    user: User
    project: Project
    object_attributes: ObjectAttributes


class NoteEvent(TypedDict, total=False):
    """A `note` delivery — a comment on an issue or merge request."""

    object_kind: str
    event_type: str
    user: User
    project: Project
    object_attributes: ObjectAttributes
    merge_request: MergeRequest
    issue: dict[str, Any]


class MergeRequest(TypedDict, total=False):
    """`payload["merge_request"]` or `payload["issue"]` — the parent of a note."""

    id: int
    iid: int
    title: str
    url: str


def merge_request(event: WebhookEvent) -> MergeRequestEvent:
    """Typed view of a `merge_request` delivery's payload. A cast, not validation."""
    return cast("MergeRequestEvent", event.payload)


def note(event: WebhookEvent) -> NoteEvent:
    """Typed view of a `note` delivery's payload. A cast, not validation."""
    return cast("NoteEvent", event.payload)
