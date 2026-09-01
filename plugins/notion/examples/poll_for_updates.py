"""Detect Notion changes with a scheduled trigger (Notion has no webhooks).

This example polls a Notion database for pages edited since the last trigger
fired and reacts to them. The trigger passes the scheduled fire time as
`start_time`, which doubles as the polling cursor: every run looks back far
enough to cover the schedule interval plus slack, and idempotency makes
overlapping windows safe.

Requirements:
    pip install flyteplugins-notion

Setup:
    flyte create secret NOTION_TOKEN --value ntn_...
    Share the target database with your integration.

Usage:
    python plugins/notion/examples/poll_for_updates.py   # deploys the trigger
"""

from datetime import datetime, timedelta, timezone

import flyte

from flyteplugins.notion import NotionClient

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-notion")

env = flyte.TaskEnvironment(
    name="notion-poll",
    image=image,
    secrets=[flyte.Secret("NOTION_TOKEN", as_env_var="NOTION_TOKEN")],
)

#: Look back slightly further than the schedule interval so nothing is missed.
LOOKBACK = timedelta(minutes=20)

poll_trigger = flyte.Trigger(
    "notion_poll_every_15m",
    flyte.FixedRate(15),
    inputs={"start_time": flyte.TriggerTime, "database_id": "REPLACE_WITH_DATABASE_ID"},
)


@env.task(triggers=(poll_trigger,))
async def poll_for_updates(start_time: datetime, database_id: str) -> int:
    """React to pages edited since the previous scheduled fire time."""
    since = (start_time.astimezone(timezone.utc) - LOOKBACK).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    async with NotionClient() as client:
        pages = await client.query_database_since.aio(database_id, since)

    for page in pages:
        # Replace with real reactions: launch a run, post to Slack, update the
        # page back, etc. Idempotent launching via flyteplugins.notion.launch_task
        # keeps overlapping windows from double-processing a page edit.
        print(f"page edited: {page['title']} ({page['url']}) at {page['last_edited_time']}")

    return len(pages)


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(env)
