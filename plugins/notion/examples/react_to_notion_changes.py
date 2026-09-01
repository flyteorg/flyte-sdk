"""React to Notion page edits with the plugin's app environment.

Notion has no webhooks, so `NotionAppEnvironment` detects changes by polling:
it serves a setup dashboard (`/`) and a poll endpoint
(`GET /api/poll?database_id=...`) that queries a database for pages edited
since a cursor and dispatches them to registered handlers.

Point any scheduler at the poll endpoint — cron, a Flyte `Trigger` making an
HTTP call, or a manual `curl` while testing. `poll_for_updates.py` shows the
webhook-free alternative: a scheduled task that calls `query_database_since`
directly, with no app in the picture at all.

Requirements:
    pip install "flyteplugins-notion[app]"

Setup:
    flyte create secret NOTION_TOKEN --value ntn_...
    flyte create secret NOTION_POLL_TOKEN --value <random-string>

    Share the target database with your integration, then deploy:
        python plugins/notion/examples/react_to_notion_changes.py

Usage:
    curl -H 'X-Poll-Token: <token>' '<app-url>/api/poll?database_id=<id>'
"""

import flyte

from flyteplugins.notion import DuplicateRun, NotionAppEnvironment, events, launch_task

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-notion[app]")

app_env = NotionAppEnvironment(
    name="notion-integration",
    image=image,
    secrets=[
        flyte.Secret("NOTION_TOKEN", as_env_var="NOTION_TOKEN"),
        flyte.Secret("NOTION_POLL_TOKEN", as_env_var="NOTION_POLL_TOKEN"),
    ],
    # Databases this app may poll. The first is the default when `/api/poll`
    # is called without a `database_id`; polling anything else is rejected.
    databases=[],
)


@app_env.on_event(events.Page.EDITED)
async def handle_page_edit(event):
    """Launch a run for each edited page.

    `dedupe_key()` folds in the page's `last_edited_time`, so overlapping poll
    windows re-reporting the same edit never launch a second run — while a
    genuinely later edit does.
    """
    import flyte.remote as remote

    task = remote.Task.get(name="write_report", auto_version="latest")
    try:
        run = await launch_task.aio(task, key=event.dedupe_key(), page_id=event.page_id)
    except DuplicateRun as exc:
        return {"skipped": str(exc)}
    return {"run": run.name}


if __name__ == "__main__":
    flyte.init_from_config()
    handle = flyte.serve(app_env)
    handle.activate(wait=True)
    print(f"Dashboard ready at {handle.endpoint}")
