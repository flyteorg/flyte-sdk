# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "slack-sdk>=3.33.5",
#     "flyte>=2.0.0b25",
# ]
# ///
"""
Slack Echo Bot - A simple bot that echoes messages in a thread until 'stop' is sent.

Setup Instructions:
1. Create a Slack App at https://api.slack.com/apps
2. Add Bot Token Scopes: chat:write, channels:history, channels:read
3. Install app to your workspace
4. Set environment variables:
   export SLACK_BOT_TOKEN="xoxb-your-token-here"
   export SLACK_CHANNEL_ID="C1234567890"

Usage:
    uv run slack_bot.py
"""

import asyncio
import os
import time
from typing import Any

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

import flyte

# Configure Flyte environment
env = flyte.TaskEnvironment(
    name="slack_echo_bot",
    image=flyte.Image.from_debian_base().with_pip_packages("slack-sdk"),
    secrets=flyte.Secret(key="slack_bot_token", as_env_var="SLACK_BOT_TOKEN"),
)


def get_slack_client() -> WebClient:
    """Initialize and return Slack WebClient."""
    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        raise ValueError("SLACK_BOT_TOKEN environment variable is required")
    return WebClient(token=token)


@flyte.trace
async def post_slack_message(channel: str, text: str, thread_ts: str | None = None) -> dict[str, Any]:
    """
    Post a message to Slack (traced for deterministic replay).

    This function is traced by Flyte, which means:
    - On crash/restart, already posted messages won't be posted again
    - Message posting is deterministic and replayable
    - Each unique (channel, text, thread_ts) combination is cached

    Args:
        channel: Channel ID to post to
        text: Message text
        thread_ts: Thread timestamp (optional, for replies)

    Returns:
        Response from Slack API
    """
    client = get_slack_client()
    try:
        response = client.chat_postMessage(channel=channel, text=text, thread_ts=thread_ts)
        return response.data
    except SlackApiError as e:
        print(f"Error posting message: {e.response['error']}")
        raise


def get_thread_replies(client: WebClient, channel: str, thread_ts: str) -> list[dict[str, Any]]:
    """
    Get all replies in a thread.

    Args:
        client: Slack WebClient instance
        channel: Channel ID
        thread_ts: Thread timestamp

    Returns:
        List of message dictionaries (excluding the parent message)
    """
    try:
        response = client.conversations_replies(channel=channel, ts=thread_ts)
        # First message is the parent, skip it
        messages = response.data.get("messages", [])
        return messages[1:] if len(messages) > 1 else []
    except SlackApiError as e:
        print(f"Error fetching replies: {e.response['error']}")
        return []


def find_last_unprocessed_message(replies: list[dict[str, Any]], bot_user_id: str) -> int:
    """
    Find the index of the last user message that hasn't been echoed by the bot.

    Strategy: Look for the last message from a user that doesn't have a bot reply immediately after it.

    Args:
        replies: List of thread replies
        bot_user_id: Bot's user ID

    Returns:
        Index to start processing from (0 if no messages processed yet)
    """
    if not replies:
        return 0

    # Start from the end and work backwards
    for i in range(len(replies) - 1, -1, -1):
        msg = replies[i]

        # If this is a bot message, skip it
        if is_bot_message(msg, bot_user_id):
            continue

        # This is a user message - check if the next message (if exists) is from the bot
        if i + 1 < len(replies):
            next_msg = replies[i + 1]
            if is_bot_message(next_msg, bot_user_id):
                # Bot already responded to this message, keep looking back
                continue

        # Found a user message without a bot response - start from here
        return i

    # All messages have been processed
    return len(replies)


def should_stop(message_text: str) -> bool:
    """Check if message contains stop command (case-insensitive)."""
    return message_text.strip().lower() == "stop"


def is_bot_message(message: dict[str, Any], bot_user_id: str) -> bool:
    """Check if message is from the bot itself."""
    return message.get("user") == bot_user_id or message.get("bot_id") is not None


@flyte.trace
async def post_greeting_message(channel_id: str, initial_message: str) -> dict[str, str]:
    """
    Post the initial greeting message to Slack.

    This function is traced by Flyte, which means:
    - If the bot crashes and restarts, it won't post the greeting again
    - Multiple bot instances will each create their own threads
    - The thread info is recoverable from the trace

    Args:
        channel_id: Slack channel ID
        initial_message: Message text to post

    Returns:
        Dict with thread_ts and thread_url
    """
    print(f"📤 Posting initial message to channel {channel_id}...")
    client = get_slack_client()
    try:
        resp = client.chat_postMessage(channel=channel_id, text=initial_message)
        initial_response = resp.data
        thread_ts = initial_response["ts"]
        thread_url = f"https://slack.com/app_redirect?channel={channel_id}&message_ts={thread_ts}"

        print(f"✅ Message posted! Thread URL: {thread_url}")

        return {
            "thread_ts": thread_ts,
            "thread_url": thread_url,
        }
    except SlackApiError as e:
        print(f"Error posting message: {e.response['error']}")
        raise


@env.task(timeout=600)
async def slack_echo_bot(
    channel_id: str,
    initial_message: str = "Hi! I'm an echo bot. Reply to this thread and I'll echo back. Send 'stop' to end.",
    poll_interval: float = 2.0,
) -> dict[str, Any]:
    """
    Slack echo bot that replies to messages in a thread until 'stop' is received.

    The task has a timeout configured in the TaskEnvironment (default: 1 hour).
    Flyte will automatically kill the task if it exceeds the timeout.

    Args:
        channel_id: Slack channel ID (required)
        initial_message: Initial message to post
        poll_interval: Seconds between polling for new messages

    Returns:
        Summary dict with message count and thread URL
    """

    # Initialize Slack client
    client = get_slack_client()

    # Get bot's user ID to filter out own messages
    auth_response = client.auth_test()
    bot_user_id = auth_response.data["user_id"]
    print(f"🤖 Bot authenticated as: {auth_response.data['user']}")

    # Post initial greeting (traced - won't repost on crash/restart)
    greeting_info = await post_greeting_message(channel_id, initial_message)
    thread_ts = greeting_info["thread_ts"]
    thread_url = greeting_info["thread_url"]

    print(f"👂 Listening for replies (polling every {poll_interval}s)...\n")

    # Track stats
    message_count = 0
    start_time = time.time()

    try:
        while True:
            # Fetch thread replies
            replies = get_thread_replies(client, channel_id, thread_ts)

            # Find where to start processing (handles crash recovery)
            start_idx = find_last_unprocessed_message(replies, bot_user_id)

            # Process messages from the last unprocessed one onwards
            for i in range(start_idx, len(replies)):
                message = replies[i]

                # Skip bot messages
                if is_bot_message(message, bot_user_id):
                    continue

                message_text = message.get("text", "")
                user = message.get("user", "unknown")

                print(f"💬 New message from <@{user}>: {message_text}")

                # Check for stop command
                if should_stop(message_text):
                    print("🛑 Stop command received! Sending goodbye...")
                    await post_slack_message(channel_id, "Bye! 👋", thread_ts)
                    print("✅ Goodbye sent. Exiting...")
                    return {
                        "status": "stopped_by_user",
                        "messages_echoed": message_count,
                        "thread_url": thread_url,
                        "runtime_seconds": int(time.time() - start_time),
                    }

                # Echo the message back (traced - won't duplicate on crash)
                echo_text = f"Echo: {message_text}"
                await post_slack_message(channel_id, echo_text, thread_ts)
                message_count += 1
                print(f"🔊 Echoed back: {echo_text}\n")

            # Wait before next poll
            await asyncio.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user. Exiting...")
        return {
            "status": "interrupted",
            "messages_echoed": message_count,
            "thread_url": thread_url,
            "runtime_seconds": int(time.time() - start_time),
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    # Initialize Flyte
    flyte.init_from_config()

    chan = os.getenv("SLACK_CHANNEL_ID")
    if not chan:
        raise ValueError("SLACK_CHANNEL_ID environment variable is required")

    # Run the bot
    run = flyte.run(
        slack_echo_bot,
        channel_id=chan,
        # You can override defaults here:
        # initial_message="Custom message here",
        # poll_interval=3.0,
    )

    print(f"Run URL: {run.url}")
