"""React to Slack events (mentions, emoji) with the plugin's app environment.

`SlackAppEnvironment` serves a setup dashboard (`/`) and an Events API
receiver (`/events`) that answers Slack's URL verification challenge and
verifies request signatures. This example launches an idempotent run when the
bot is mentioned, and acknowledges `:bug:` reactions.

Requirements:
    pip install "flyteplugins-slack[app]"

Setup:
    flyte create secret SLACK_BOT_TOKEN --value xoxb-...
    flyte create secret SLACK_SIGNING_SECRET --value <signing-secret>

    Deploy the app, then in your Slack app's Event Subscriptions set the
    Request URL to `<app-url>/events` (the challenge handshake is automatic)
    and subscribe to `app_mention` and `reaction_added`.

Usage:
    python plugins/slack/examples/react_to_slack_events.py
"""

import flyte

from flyteplugins.slack import SlackAppEnvironment, SlackClient, launch_task

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-slack[app]")

app_env = SlackAppEnvironment(
    name="slack-integration",
    image=image,
    secrets=[
        flyte.Secret("SLACK_BOT_TOKEN", as_env_var="SLACK_BOT_TOKEN"),
        flyte.Secret("SLACK_SIGNING_SECRET", as_env_var="SLACK_SIGNING_SECRET"),
    ],
    # Only react in these channel ids (empty = all channels the bot is in).
    channels=[],
)


@app_env.on_event("app_mention")
async def answer_mention(event):
    """Launch an agent run once per mention thread.

    The `answer_mention` task must already be deployed. `launch_task` dedupes
    on channel + thread root, so Slack retries never launch a second run.
    """
    import flyte.remote as remote

    from flyteplugins.slack import DuplicateRun

    task = remote.Task.get(name="answer_mention", auto_version="latest")
    try:
        run = launch_task(
            task,
            key=event.dedupe_key(),
            channel=event.channel,
            thread_ts=event.root_ts,
            question=event.text,
        )
    except DuplicateRun as exc:
        return {"skipped": str(exc)}
    return {"run": run.name}


@app_env.on_event("reaction_added")
async def acknowledge_bug_reaction(event):
    """React to a :bug: emoji by acknowledging it in-thread."""
    if event.reaction != "bug":
        return None
    async with SlackClient() as client:
        await client.reply_in_thread(event.channel, event.root_ts, ":eyes: Flyte saw the :bug: — investigating.")
    return {"acknowledged": event.root_ts}


if __name__ == "__main__":
    flyte.init_from_config()
    handle = flyte.serve(app_env)
    handle.activate(wait=True)
    print(f"Dashboard ready at {handle.endpoint}")
