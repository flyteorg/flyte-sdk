"""Slack + Email notification example.

Sends a rich Slack notification and an email when a task succeeds or fails.

Usage:
    export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    export NOTIFICATION_EMAIL="oncall@example.com"
    python examples/notification/slack_webhook.py

To receive emails locally, start a debug SMTP server before running:

    # Python < 3.12
    sudo python -m smtpd -n -c DebuggingServer localhost:25

    # Python >= 3.12 (smtpd was removed; use aiosmtpd)
    pip install aiosmtpd
    sudo python -m aiosmtpd -n -l localhost:25

Both print received emails to stdout. Port 25 requires root (sudo);
alternatively use port 1025, but update the SMTP port in _sender.py.
"""

import os

import flyte
from flyte import notify
from flyte.models import ActionPhase

env = flyte.TaskEnvironment(name="notify_example")

SLACK_WEBHOOK_URL = os.environ["SLACK_WEBHOOK_URL"]
NOTIFICATION_EMAIL = os.environ["NOTIFICATION_EMAIL"]


@env.task
def compute(x: int, y: int) -> int:
    return x + y


slack_success = notify.Slack(
    on_phase=ActionPhase.SUCCEEDED,
    webhook_url=SLACK_WEBHOOK_URL,
    blocks=[
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Task Succeeded"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": "*Task:*\n{task.name}"},
                {"type": "mrkdwn", "text": "*Run:*\n{run.name}"},
                {"type": "mrkdwn", "text": "*Duration:*\n{run.duration}"},
                {"type": "mrkdwn", "text": "*Phase:*\n{run.phase}"},
            ],
        },
        {"type": "divider"},
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": "{project}/{domain}"},
            ],
        },
    ],
)

slack_failure = notify.Slack(
    on_phase=ActionPhase.FAILED,
    webhook_url=SLACK_WEBHOOK_URL,
    blocks=[
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Task Failed"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": "*Task:*\n{task.name}"},
                {"type": "mrkdwn", "text": "*Error:*\n{run.error}"},
            ],
        },
    ],
)

email_success = notify.Email(
    on_phase=ActionPhase.SUCCEEDED,
    recipients=[NOTIFICATION_EMAIL],
    subject="Task {task.name} succeeded",
    body=("Task: {task.name}\nRun: {run.name}\nDuration: {run.duration}\nPhase: {run.phase}\n"),
)

email_failure = notify.Email(
    on_phase=ActionPhase.FAILED,
    recipients=[NOTIFICATION_EMAIL],
    subject="ALERT: Task {task.name} failed",
    body=("Task: {task.name}\nRun: {run.name}\nError: {run.error}\n"),
)

if __name__ == "__main__":
    result = flyte.with_runcontext(
        mode="local",
        notifications=(slack_success, slack_failure, email_success, email_failure),
    ).run(compute, x=3, y=7)
    print(f"Result: {result}")
