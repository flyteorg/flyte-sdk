"""One app receiving webhooks from every provider, launching tasks per event.

This is the whole point of `flyteplugins-webhooks`: authenticate an inbound
delivery, normalize it, and turn it into a run. The tasks it launches live in
the sibling examples here and talk to each product through that product's own
maintained SDK — no Flyte plugin sits between them and the API.

Requirements:
    pip install "flyteplugins-github[app]" flyteplugins-slack  # ...and any others

Setup:
    Store the secret for each provider you enable. The app mounts them from each
    provider's default_secret_env, so they need no repeating in `secrets=`:
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

from flyteplugins.clickup import ClickUpProvider
from flyteplugins.clickup import events as clickup_events
from flyteplugins.github import GitHubProvider
from flyteplugins.github import events as github_events
from flyteplugins.jira import JiraProvider
from flyteplugins.jira import events as jira_events
from flyteplugins.linear import LinearProvider
from flyteplugins.linear import events as linear_events
from flyteplugins.slack import SlackProvider
from flyteplugins.slack import events as slack_events

import flyte
from flyte.extras.webhooks import DuplicateRun, WebhookAppEnvironment, idempotent_run

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    # Each product is its own package; the receiver itself ships with flyte.
    "flyteplugins-github[app]",
    "flyteplugins-slack",
    "flyteplugins-linear",
    "flyteplugins-clickup",
    "flyteplugins-jira",
)

app_env = WebhookAppEnvironment(
    name="saas-webhooks",
    # Each provider is its own package; install and list only the ones you
    # wire up. Anything not listed 404s, and each one's secret is mounted for
    # you from its default_secret_env.
    providers=[GitHubProvider(), SlackProvider(), LinearProvider(), ClickUpProvider(), JiraProvider()],
    image=image,
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


@app_env.on_event(github_events.PullRequest.OPENED)
async def triage_new_pr(event):
    """Triage every newly opened pull request."""
    repo, _, number = (event.resource_id or "").partition("#")
    return await _launch("github-triage.triage_pr", event, repo=repo, number=int(number))


@app_env.on_event(slack_events.AppMention.ANY)
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


@app_env.on_event(linear_events.Issue.CREATE)
async def triage_new_linear_issue(event):
    return await _launch("linear-triage.triage_issue", event, issue_id=event.resource_id)


@app_env.on_event(clickup_events.Task.CREATED)
async def triage_new_clickup_task(event):
    return await _launch("clickup-tickets.triage_task", event, task_id=event.resource_id)


@app_env.on_event(jira_events.Issue.CREATED)
async def triage_new_jira_issue(event):
    return await _launch("jira-tickets.triage_issue", event, issue_key=event.resource_id)


if __name__ == "__main__":
    flyte.init_from_config()
    handle = flyte.serve(app_env)
    handle.activate(wait=True)
    print(f"Dashboard ready at {handle.endpoint}")
