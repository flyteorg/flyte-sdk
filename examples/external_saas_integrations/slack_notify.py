"""Post to Slack from tasks, with the official `slack_sdk`.

Requirements:
    pip install flyte slack_sdk

Setup:
    flyte create secret SLACK_BOT_TOKEN --value xoxb-...
    Invite the bot to the channel: /invite @your-app

Usage:
    flyte run examples/external_saas_integrations/slack_notify.py \\
        notify --channel C01234567 --message "Hello from Flyte!"
"""

import os

import flyte

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("slack_sdk")

env = flyte.TaskEnvironment(
    name="slack-notify",
    image=image,
    secrets=[flyte.Secret("SLACK_BOT_TOKEN", as_env_var="SLACK_BOT_TOKEN")],
)


def _client():
    from slack_sdk.web.async_client import AsyncWebClient

    return AsyncWebClient(token=os.environ["SLACK_BOT_TOKEN"])


@env.task
async def notify(channel: str, message: str) -> str:
    """Post a message and return its permalink."""
    client = _client()
    posted = await client.chat_postMessage(channel=channel, text=message)
    link = await client.chat_getPermalink(channel=channel, message_ts=posted["ts"])
    return link["permalink"]


@env.task
async def answer_mention(channel: str, thread_ts: str, question: str) -> str:
    """Reply in the thread that mentioned the bot.

    This is what `webhook_receiver.py` launches for every mention. Swap the
    canned reply for an agent call to make it do real work.
    """
    client = _client()
    await client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=f"Got it — working on: {question}")
    return f"answered {channel}/{thread_ts}"


@env.task
async def progress_thread(channel: str, status: str) -> str:
    """Start a thread, post progress in it, and react to the root."""
    client = _client()
    root = await client.chat_postMessage(channel=channel, text=f":rocket: {status} started")
    await client.chat_postMessage(channel=channel, thread_ts=root["ts"], text="Working on it...")
    await client.chat_postMessage(channel=channel, thread_ts=root["ts"], text=f":white_check_mark: {status} finished")
    await client.reactions_add(channel=channel, timestamp=root["ts"], name="white_check_mark")
    return str(root["ts"])


if __name__ == "__main__":
    flyte.init_from_config()
    # Replace with a channel id your bot is a member of.
    print(flyte.run(notify, channel="C01234567", message="Hello from Flyte!").url)
