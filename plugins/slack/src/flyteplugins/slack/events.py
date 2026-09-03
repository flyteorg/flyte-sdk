"""Slack Events API events. `message` carries subtypes; the rest are bare types."""

from __future__ import annotations

from flyte.extras.webhooks import EventType

__all__ = ["AppHome", "AppMention", "Channel", "File", "Member", "Message", "Pin", "Reaction", "Team"]


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
