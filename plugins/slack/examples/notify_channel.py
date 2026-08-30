"""Send Slack messages and react-style updates from Flyte tasks.

This example shows the basic client surface: posting messages, replying in
threads, updating messages as a run progresses, and adding reactions.

Requirements:
    pip install flyteplugins-slack

Setup:
    flyte create secret SLACK_BOT_TOKEN --value xoxb-...

Usage:
    python plugins/slack/examples/notify_channel.py
"""

import flyte

from flyteplugins.slack import SlackClient

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-slack")

env = flyte.TaskEnvironment(
    name="slack-notify",
    image=image,
    secrets=[flyte.Secret("SLACK_BOT_TOKEN", as_env_var="SLACK_BOT_TOKEN")],
)


@env.task
async def notify(channel: str, message: str) -> str:
    """Post a message and return its permalink."""
    async with SlackClient() as client:
        result = await client.post_message(channel, message)
    return result.get("permalink", result.get("ts", ""))


@env.task
async def progress_thread(channel: str, status: str) -> str:
    """Start a thread, post progress updates in it, and react to the root."""
    async with SlackClient() as client:
        root = await client.post_message(channel, f":rocket: {status} started")
        await client.reply_in_thread(channel, root["ts"], "Working on it...")
        await client.reply_in_thread(channel, root["ts"], f":white_check_mark: {status} finished")
        await client.add_reaction(channel, root["ts"], "flyte")
    return root.get("permalink", root["ts"])


if __name__ == "__main__":
    # Replace with a channel id your bot is a member of.
    flyte.run(notify, channel="C01234567", message="Hello from Flyte!")
