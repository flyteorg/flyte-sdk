"""One app receiving webhooks from every provider, launching tasks per event.

This is the whole point of `flyteplugins-webhooks`: authenticate an inbound
delivery, normalize it, and turn it into a run. The tasks it launches live in
the sibling examples here and talk to each product through that product's own
maintained SDK — no Flyte plugin sits between them and the API.

Requirements:
    pip install "flyteplugins-webhooks[app]"

Setup:
    Store the secret for each provider you enable:
        flyte create secret GITHUB_WEBHOOK_SECRET --value <random-string>
        flyte create secret SLACK_SIGNING_SECRET  --value <from Slack app>
        flyte create secret LINEAR_WEBHOOK_SECRET --value <from Linear webhook>
        flyte create secret CLICKUP_WEBHOOK_SECRET --value <from ClickUp webhook>
        flyte create secret JIRA_WEBHOOK_TOKEN    --value <random-string>

    Deploy the tasks first — the handlers look them up by name:
        flyte deploy examples/external_saas_integrations/github_triage_pr.py env
        flyte deploy examples/external_saas_integrations/slack_notify.py env

Usage:
    python examples/external_saas_integrations/webhook_receiver.py

    Then point each provider at the URL the dashboard shows for it.
"""

from flyteplugins.webhooks import WebhookAppEnvironment, events

import flyte
from flyte.extras import DuplicateRun, idempotent_run

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-webhooks[app]")

app_env = WebhookAppEnvironment(
    name="saas-webhooks",
    # Drop any provider you are not wiring up; its route then 404s.
    providers=["github", "slack", "linear", "clickup", "jira"],
    image=image,
    secrets=[
        flyte.Secret("GITHUB_WEBHOOK_SECRET", as_env_var="GITHUB_WEBHOOK_SECRET"),
        flyte.Secret("SLACK_SIGNING_SECRET", as_env_var="SLACK_SIGNING_SECRET"),
        flyte.Secret("LINEAR_WEBHOOK_SECRET", as_env_var="LINEAR_WEBHOOK_SECRET"),
        flyte.Secret("CLICKUP_WEBHOOK_SECRET", as_env_var="CLICKUP_WEBHOOK_SECRET"),
        flyte.Secret("JIRA_WEBHOOK_TOKEN", as_env_var="JIRA_WEBHOOK_TOKEN"),
    ],
    # Only react to events from these repos / channels / teams / lists /
    # projects. Empty means all; an allowlist also drops events it cannot
    # attribute to a scope.
    scopes=[],
)


async def _launch(task_name: str, event, **inputs):
    """Look up a deployed task and launch it once for this event.

    `task_name` is the deployed name, which Flyte qualifies with the task
    environment: `<env-name>.<task_name>`. A bare function name never resolves,
    and two examples here both define `triage_issue` — the qualifier keeps them
    apart.

    Always `await idempotent_run.aio(...)`. The blocking form stalls the app's
    event loop, and webhook senders time deliveries out in seconds.
    """
    import flyte.remote as remote

    task = remote.Task.get(name=task_name, auto_version="latest")
    try:
        run = await idempotent_run.aio(task, key=event.dedupe_key(), **inputs)
    except DuplicateRun as exc:
        return {"skipped": str(exc)}
    return {"run": run.name}


@app_env.on_event(events.github.PullRequest.OPENED)
async def triage_new_pr(event):
    """Triage every newly opened pull request."""
    repo, _, number = (event.resource_id or "").partition("#")
    return await _launch("github-triage.triage_pr", event, repo=repo, number=int(number))


@app_env.on_event(events.slack.AppMention.ANY)
async def answer_mention(event):
    """Answer every mention of the bot.

    The default key is per message, so each mention gets its own run. For one
    run per *thread* instead, pass your own key built from `thread_ts`.
    """
    slack_event = event.payload.get("event", {})
    return await _launch(
        "slack-notify.answer_mention",
        event,
        channel=event.scope,
        thread_ts=slack_event.get("thread_ts") or slack_event.get("ts"),
        question=slack_event.get("text", ""),
    )


@app_env.on_event(events.linear.Issue.CREATE)
async def triage_new_linear_issue(event):
    return await _launch("linear-triage.triage_issue", event, issue_id=event.resource_id)


@app_env.on_event(events.clickup.Task.CREATED)
async def triage_new_clickup_task(event):
    return await _launch("clickup-tickets.triage_task", event, task_id=event.resource_id)


@app_env.on_event(events.jira.Issue.CREATED)
async def triage_new_jira_issue(event):
    return await _launch("jira-tickets.triage_issue", event, issue_key=event.resource_id)


if __name__ == "__main__":
    flyte.init_from_config()
    handle = flyte.serve(app_env)
    handle.activate(wait=True)
    print(f"Dashboard ready at {handle.endpoint}")
