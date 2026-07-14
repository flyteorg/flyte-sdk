"""Combined notification example — fan out to every channel at once.

A single run can carry multiple notifications across different channels and
phases. Here we wire up Slack, Email, Teams, and a custom Webhook together so
that one run can alert several systems depending on how it finishes.

Usage:
    export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    export TEAMS_WEBHOOK_URL="https://outlook.office.com/webhook/YOUR/WEBHOOK/URL"
    export ALERT_WEBHOOK_URL="https://webhook.site/your-unique-id"
    export NOTIFICATION_EMAIL="oncall@example.com"
    python examples/notification/all_channels.py
"""

import os

import flyte
from flyte import notify
from flyte.models import ActionPhase
from flyte.remote import Run

env = flyte.TaskEnvironment(
    name="notify_all_channels",
    resources=flyte.Resources(memory="250Mi"),
)

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/REPLACE/ME")
TEAMS_WEBHOOK_URL = os.environ.get("TEAMS_WEBHOOK_URL", "https://outlook.office.com/webhook/REPLACE/ME")
ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "https://webhook.site/replace-me")
NOTIFICATION_EMAIL = os.environ.get("NOTIFICATION_EMAIL", "oncall@example.com")

# Phases that count as "the run is done", regardless of outcome.
ALL_TERMINAL = (ActionPhase.SUCCEEDED, ActionPhase.FAILED, ActionPhase.ABORTED, ActionPhase.TIMED_OUT)


@env.task
def compute(x: int, y: int) -> int:
    return x + y


notifications = (
    # Slack: ping the team channel on success.
    notify.Slack(
        on_phase=ActionPhase.SUCCEEDED,
        webhook_url=SLACK_WEBHOOK_URL,
        message="✅ Run {{.Run.Name}} succeeded in {{.Run.Project}}/{{.Run.Domain}}",
    ),
    # Email: page on-call when something goes wrong.
    notify.Email(
        on_phase=(ActionPhase.FAILED, ActionPhase.TIMED_OUT),
        recipients=(NOTIFICATION_EMAIL,),
        subject="🚨 Run {{.Run.Name}} failed ({{.Phase}})",
        body="Run: {{.Run.Name}}\nPhase: {{.Phase}}\nError: {{.Error}}\n",
    ),
    # Teams: mirror the failure alert into Teams.
    notify.Teams(
        on_phase=(ActionPhase.FAILED, ActionPhase.TIMED_OUT),
        webhook_url=TEAMS_WEBHOOK_URL,
        title="🚨 Run {{.Run.Name}} failed",
        message="Phase {{.Phase}}: {{.Error}}",
    ),
    # Webhook: record every terminal outcome to a custom endpoint.
    notify.Webhook(
        on_phase=ALL_TERMINAL,
        url=ALERT_WEBHOOK_URL,
        headers={"Content-Type": "application/json"},
        body={
            "run": "{{.Run.Name}}",
            "phase": "{{.Phase}}",
            "error": "{{.Error}}",
        },
    ),
)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(
        env_vars={
            "SLACK_WEBHOOK_URL": SLACK_WEBHOOK_URL,
            "TEAMS_WEBHOOK_URL": TEAMS_WEBHOOK_URL,
            "ALERT_WEBHOOK_URL": ALERT_WEBHOOK_URL,
            "NOTIFICATION_EMAIL": NOTIFICATION_EMAIL,
        },
        notifications=notifications,
    ).run(compute, x=3, y=7)
    assert isinstance(run, Run)

    print(run.name)
    print(run.url)
    run.wait()
