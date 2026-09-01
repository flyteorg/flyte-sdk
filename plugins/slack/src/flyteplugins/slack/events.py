"""Typed constants for the Slack Events API events an app can subscribe to.

Register handlers with these instead of raw strings, so an editor can complete
them and a typo fails at import rather than by silently never matching:

```python
from flyteplugins.slack import SlackAppEnvironment, events

app_env = SlackAppEnvironment(name="slack-integration")

@app_env.on_event(events.AppMention.ANY)
async def handle(event): ...
```

Each class groups related Slack event types. Where an event carries a subtype
(`message` does), members spell the `type.subtype` pattern `on_event` matches
on; `ANY` is the bare event type, matching every subtype.

These are `str` subclasses, so they are drop-in wherever a pattern string is
accepted. `on_event` still takes plain strings too — reach for one when Slack
ships an event these constants do not cover yet.
"""

from __future__ import annotations

import enum

__all__ = [
    "AppHome",
    "AppMention",
    "Channel",
    "File",
    "Member",
    "Message",
    "Pin",
    "Reaction",
    "Team",
]


class _EventType(str, enum.Enum):
    """Base for event constants: a real `str`, usable anywhere a pattern is."""

    # Without these, Python 3.11+ renders members as "Class.MEMBER" in
    # f-strings and str(), rather than the wire value handlers match on.
    __str__ = str.__str__
    __format__ = str.__format__  # type: ignore[assignment]


class Message(_EventType):
    """`message` events. Members below are Slack's message subtypes."""

    ANY = "message"
    """Every message, including those carrying a subtype."""
    CHANGED = "message.message_changed"
    DELETED = "message.message_deleted"
    REPLIED = "message.message_replied"
    CHANNEL_JOIN = "message.channel_join"
    CHANNEL_LEAVE = "message.channel_leave"
    BOT_MESSAGE = "message.bot_message"
    FILE_SHARE = "message.file_share"
    THREAD_BROADCAST = "message.thread_broadcast"


class AppMention(_EventType):
    """`app_mention` events — the bot was @-mentioned. No subtype."""

    ANY = "app_mention"


class Reaction(_EventType):
    """Emoji reaction events."""

    ADDED = "reaction_added"
    REMOVED = "reaction_removed"


class Channel(_EventType):
    """Channel lifecycle events."""

    CREATED = "channel_created"
    DELETED = "channel_deleted"
    RENAME = "channel_rename"
    ARCHIVE = "channel_archive"
    UNARCHIVE = "channel_unarchive"


class Member(_EventType):
    """Channel membership events."""

    JOINED_CHANNEL = "member_joined_channel"
    LEFT_CHANNEL = "member_left_channel"


class Team(_EventType):
    """Workspace-level events."""

    JOIN = "team_join"
    """A new member joined the workspace."""


class File(_EventType):
    """File events."""

    CREATED = "file_created"
    SHARED = "file_shared"
    DELETED = "file_deleted"


class Pin(_EventType):
    """Pinned-item events."""

    ADDED = "pin_added"
    REMOVED = "pin_removed"


class AppHome(_EventType):
    """App Home events."""

    OPENED = "app_home_opened"
