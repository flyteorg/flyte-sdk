"""Slack events. `message` carries subtypes; the rest are bare types.

Beyond the Events API, `Interaction` covers interactivity payloads (Block Kit
actions, shortcuts, modals) and `Command` covers slash commands — both arrive
form-encoded on the same `/webhook/slack` route.
"""

from __future__ import annotations

from flyte.extras.webhooks import EventType

__all__ = [
    "AppHome",
    "AppMention",
    "Channel",
    "Command",
    "File",
    "Interaction",
    "Member",
    "Message",
    "Pin",
    "Reaction",
    "Team",
]


class Message(EventType):
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


class AppMention(EventType):
    """`app_mention` events — the bot was @-mentioned. No subtype."""

    ANY = "app_mention"


class Reaction(EventType):
    """Emoji reaction events."""

    ADDED = "reaction_added"
    REMOVED = "reaction_removed"


class Channel(EventType):
    """Channel lifecycle events."""

    CREATED = "channel_created"
    DELETED = "channel_deleted"
    RENAME = "channel_rename"
    ARCHIVE = "channel_archive"
    UNARCHIVE = "channel_unarchive"


class Member(EventType):
    """Channel membership events."""

    JOINED_CHANNEL = "member_joined_channel"
    LEFT_CHANNEL = "member_left_channel"


class Team(EventType):
    """Workspace-level events."""

    JOIN = "team_join"
    """A new member joined the workspace."""


class File(EventType):
    """File events."""

    CREATED = "file_created"
    SHARED = "file_shared"
    DELETED = "file_deleted"


class Pin(EventType):
    """Pinned-item events."""

    ADDED = "pin_added"
    REMOVED = "pin_removed"


class AppHome(EventType):
    """App Home events."""

    OPENED = "app_home_opened"


class Interaction(EventType):
    """Interactivity payloads: Block Kit actions, shortcuts, and modals.

    Point the app's *Interactivity & Shortcuts* Request URL at the same
    `/webhook/slack` route. The event's action is the `action_id` (or
    `callback_id`), so a constant below matches a whole payload type, and a raw
    string like `"block_actions.approve_reply"` matches one button.
    """

    BLOCK_ACTIONS = "block_actions"
    """A Block Kit interactive component was used — a button, select, overflow menu."""
    VIEW_SUBMISSION = "view_submission"
    VIEW_CLOSED = "view_closed"
    SHORTCUT = "shortcut"
    """A global shortcut. Its action is the shortcut's `callback_id`."""
    MESSAGE_ACTION = "message_action"
    """A message shortcut. Its action is the shortcut's `callback_id`."""


class Command(EventType):
    """Slash commands. A raw string like `"command.deploy"` matches `/deploy` alone."""

    ANY = "command"
