"""React to Linear webhooks with the plugin's app environment.

`LinearAppEnvironment` serves a setup dashboard (`/`) and an HMAC-verified
webhook receiver (`/webhook`). This example launches an idempotent triage run
for every newly created issue and acknowledges updates with a comment.

Requirements:
    pip install "flyteplugins-linear[app]"

Setup:
    flyte create secret LINEAR_API_KEY --value <api-key>
    flyte create secret LINEAR_WEBHOOK_SECRET --value <signing-secret>

    Deploy this app, then create a Linear webhook (Settings → API → Webhooks)
    pointing at `<app-url>/webhook` with the Issues resource, and copy its
    signing secret into the secret above.

Usage:
    python plugins/linear/examples/react_to_linear_events.py
"""

import flyte

from flyteplugins.linear import LinearAppEnvironment, LinearClient, launch_task

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-linear[app]")

app_env = LinearAppEnvironment(
    name="linear-integration",
    image=image,
    secrets=[
        flyte.Secret("LINEAR_API_KEY", as_env_var="LINEAR_API_KEY"),
        flyte.Secret("LINEAR_WEBHOOK_SECRET", as_env_var="LINEAR_WEBHOOK_SECRET"),
    ],
    # Only react to events from these team ids (empty = all teams).
    team_ids=[],
)


@app_env.on_event("Issue.create")
async def triage_new_issue(event):
    """Launch the triage task once per new issue.

    The `triage_issue` task must already be deployed (see
    examples/triage_issue.py). `launch_task` dedupes on the event, so webhook
    redeliveries never launch a second run.
    """
    import flyte.remote as remote

    from flyteplugins.linear import DuplicateRun

    task = remote.Task.get(name="triage_issue", auto_version="latest")
    try:
        run = await launch_task.aio(task, key=event.dedupe_key(), issue_id=event.entity_id)
    except DuplicateRun as exc:
        return {"skipped": str(exc)}
    return {"run": run.name}


@app_env.on_event("Issue.update")
async def note_state_changes(event):
    """Comment when an issue moves into a Done-like state."""
    if event.state_id is None:
        return None
    async with LinearClient() as client:
        issue = await client.get_issue(event.entity_id)
        if issue["state"] in ("Done", "Canceled", "Duplicate"):
            await client.add_comment(event.entity_id, f"Flyte noticed this issue is now {issue['state']}.")
            return {"noted": issue["identifier"]}
    return None


if __name__ == "__main__":
    flyte.init_from_config()
    handle = flyte.serve(app_env)
    handle.activate(wait=True)
    print(f"Dashboard ready at {handle.endpoint}")
