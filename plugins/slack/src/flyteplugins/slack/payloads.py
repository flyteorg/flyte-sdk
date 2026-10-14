"""Typed views of Slack payloads, for autocomplete inside handlers.

`event.payload` is `dict[str, Any]` — correct, since it carries Slack's JSON
verbatim, but blind to write against. These TypedDicts spell the fields Slack
actually sends, so `payloads.block_actions(event)` gives editors and agents
something to complete against:

```python
from flyteplugins.slack import payloads

@app_env.on_event(events.Interaction.BLOCK_ACTIONS, action="approve_deploy")
async def on_approval(event):
    payload = payloads.block_actions(event)
    clicked = payload["actions"][0]["value"]
    where = payload["container"]["channel_id"]
```

The helpers are casts, not validators: the dict is returned untouched, and a
field Slack did not send is still a `KeyError` at runtime. Every class is
`total=False` because Slack omits fields freely — treat presence the way you
already would with a raw dict. Fields beyond these still exist in the dict;
the types name the commonly-read ones, not the whole wire format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, cast

if TYPE_CHECKING:
    from flyte.extras.webhooks import WebhookEvent

__all__ = [
    "BlockAction",
    "BlockActionsPayload",
    "Channel",
    "CommandPayload",
    "Container",
    "Message",
    "Team",
    "User",
    "block_actions",
    "command",
]


class User(TypedDict, total=False):
    """Who interacted — `payload["user"]`."""

    id: str
    username: str
    name: str
    team_id: str


class Team(TypedDict, total=False):
    id: str
    domain: str


class Channel(TypedDict, total=False):
    id: str
    name: str


class Container(TypedDict, total=False):
    """Where the interacted-with message lives — the reply/update address."""

    type: str
    message_ts: str
    channel_id: str
    thread_ts: str
    is_ephemeral: bool


class BlockAction(TypedDict, total=False):
    """One element of `payload["actions"]` — which control was used, and how."""

    action_id: str
    block_id: str
    value: str
    action_ts: str
    type: str
    text: dict[str, Any]
    selected_option: dict[str, Any]
    selected_options: list[dict[str, Any]]
    selected_date: str
    selected_user: str


class Message(TypedDict, total=False):
    """The message carrying the clicked element, when there is one."""

    ts: str
    thread_ts: str
    text: str
    user: str
    blocks: list[dict[str, Any]]


class BlockActionsPayload(TypedDict, total=False):
    """An interactivity payload: Block Kit actions, and the shared envelope
    of shortcuts and modal submissions (those carry `callback_id`/`view`)."""

    type: str
    trigger_id: str
    user: User
    team: Team
    channel: Channel
    container: Container
    actions: list[BlockAction]
    message: Message
    view: dict[str, Any]
    callback_id: str
    action_ts: str
    response_url: str
    api_app_id: str
    token: str


class CommandPayload(TypedDict, total=False):
    """A slash command's form fields, flat by design."""

    command: str
    text: str
    channel_id: str
    channel_name: str
    user_id: str
    user_name: str
    team_id: str
    team_domain: str
    trigger_id: str
    response_url: str
    api_app_id: str
    token: str


def block_actions(event: WebhookEvent) -> BlockActionsPayload:
    """Typed view of an interactivity delivery's payload. A cast, not validation."""
    return cast("BlockActionsPayload", event.payload)


def command(event: WebhookEvent) -> CommandPayload:
    """Typed view of a slash command's payload. A cast, not validation."""
    return cast("CommandPayload", event.payload)
