"""React to ClickUp webhooks with the plugin's app environment.

`ClickUpAppEnvironment` serves a setup dashboard (`/`) and an HMAC-verified
webhook receiver (`/webhook`). This example launches an idempotent run for
every newly created task and mirrors status changes back as comments.

Requirements:
    pip install "flyteplugins-clickup[app]"

Setup:
    flyte create secret CLICKUP_TOKEN --value <token>
    flyte create secret CLICKUP_WEBHOOK_SECRET --value <signing-secret>

    Deploy this app, then add a webhook in ClickUp (space/list settings →
    Webhooks) pointing at `<app-url>/webhook`, and copy its signing secret
    into the secret above.

Usage:
    python plugins/clickup/examples/react_to_clickup_events.py
"""

import flyte

from flyteplugins.clickup import ClickUpAppEnvironment, ClickUpClient, launch_task

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-clickup[app]")

app_env = ClickUpAppEnvironment(
    name="clickup-integration",
    image=image,
    secrets=[
        flyte.Secret("CLICKUP_TOKEN", as_env_var="CLICKUP_TOKEN"),
        flyte.Secret("CLICKUP_WEBHOOK_SECRET", as_env_var="CLICKUP_WEBHOOK_SECRET"),
    ],
    # Only react to events from these list ids (empty = all lists).
    list_ids=[],
)


@app_env.on_event("taskCreated")
async def triage_new_task(event):
    """Launch the triage task once per new ClickUp task.

    The `triage_task` task must already be deployed (see
    examples/manage_ticket.py for the kind of task to register). `launch_task`
    dedupes on the event, so webhook redeliveries never launch a second run.
    """
    import flyte.remote as remote

    from flyteplugins.clickup import DuplicateRun

    task = remote.Task.get(name="triage_task", auto_version="latest")
    try:
        run = await launch_task.aio(task, key=event.dedupe_key(), task_id=event.task_id)
    except DuplicateRun as exc:
        return {"skipped": str(exc)}
    return {"run": run.name}


@app_env.on_event("taskStatusUpdated")
async def note_status_changes(event):
    """Comment when a task reaches a Done-like status."""
    if event.task_status not in ("done", "complete", "closed"):
        return None
    async with ClickUpClient() as client:
        await client.add_comment(event.task_id, f"Flyte noticed this task is now {event.task_status}.")
    return {"noted": event.task_id}


if __name__ == "__main__":
    flyte.init_from_config()
    handle = flyte.serve(app_env)
    handle.activate(wait=True)
    print(f"Dashboard ready at {handle.endpoint}")
