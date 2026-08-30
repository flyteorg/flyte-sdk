"""React to Jira webhooks with the plugin's app environment.

`JiraAppEnvironment` serves a setup dashboard (`/`) and a webhook receiver
(`/webhook`) protected by a shared `X-Webhook-Token` header (Jira webhooks are
not signed). This example launches an idempotent run for every newly created
issue and comments when an issue transitions to Done.

Requirements:
    pip install "flyteplugins-jira[app]"

Setup:
    flyte create secret JIRA_BASE_URL --value https://<site>.atlassian.net
    flyte create secret JIRA_EMAIL --value you@example.com
    flyte create secret JIRA_API_TOKEN --value <api-token>
    flyte create secret JIRA_WEBHOOK_TOKEN --value <random-string>

    Deploy this app, then create a Jira webhook (gear → Products → Webhooks)
    pointing at `<app-url>/webhook`, delivered through a gateway that adds the
    `X-Webhook-Token` header. The dashboard explains the token options.

Usage:
    python plugins/jira/examples/react_to_jira_events.py
"""

import flyte

from flyteplugins.jira import JiraAppEnvironment, JiraClient, launch_task

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-jira[app]")

app_env = JiraAppEnvironment(
    name="jira-integration",
    image=image,
    secrets=[
        flyte.Secret("JIRA_BASE_URL", as_env_var="JIRA_BASE_URL"),
        flyte.Secret("JIRA_EMAIL", as_env_var="JIRA_EMAIL"),
        flyte.Secret("JIRA_API_TOKEN", as_env_var="JIRA_API_TOKEN"),
        flyte.Secret("JIRA_WEBHOOK_TOKEN", as_env_var="JIRA_WEBHOOK_TOKEN"),
    ],
    # Only react to events from these project keys (empty = all projects).
    project_keys=[],
)


@app_env.on_event("jira:issue_created")
async def triage_new_issue(event):
    """Launch the triage task once per new issue.

    The `triage_issue` task must already be deployed (see
    examples/manage_ticket.py for the kind of task to register). `launch_task`
    dedupes on the event, so webhook redeliveries never launch a second run.
    """
    import flyte.remote as remote

    from flyteplugins.jira import DuplicateRun

    task = remote.Task.get(name="triage_issue", auto_version="latest")
    try:
        run = launch_task(task, key=event.dedupe_key(), issue_key=event.issue_key)
    except DuplicateRun as exc:
        return {"skipped": str(exc)}
    return {"run": run.name}


@app_env.on_event("jira:issue_updated")
async def note_done_transitions(event):
    """Comment when an issue reaches Done."""
    if event.status != "Done":
        return None
    async with JiraClient() as client:
        await client.add_comment(event.issue_key, "Flyte noticed this issue is now Done.")
    return {"noted": event.issue_key}


if __name__ == "__main__":
    flyte.init_from_config()
    handle = flyte.serve(app_env)
    handle.activate(wait=True)
    print(f"Dashboard ready at {handle.endpoint}")
