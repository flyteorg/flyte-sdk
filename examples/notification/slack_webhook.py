"""Slack notification example.

Sends a rich Slack notification (using Block Kit) when a run reaches a terminal
phase. A simpler text-only Slack notification is also shown for failures.

Usage:
    export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    python examples/notification/slack_webhook.py

Template variables available in messages:
    {{.Run.Name}}, {{.Run.Project}}, {{.Run.Domain}}, {{.Phase}}, {{.Error}}
"""

import os

import flyte
from flyte import notify
from flyte.models import ActionPhase

env = flyte.TaskEnvironment(
    name="notify_slack",
    resources=flyte.Resources(memory="250Mi"),
)

# Fall back to a placeholder so the module imports even without the env var set.
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/REPLACE/ME")


@env.task
def compute(x: int, y: int) -> int:
    return x + y


# Rich Block Kit notification on success.
slack_success = notify.Slack(
    on_phase=ActionPhase.SUCCEEDED,
    webhook_url=SLACK_WEBHOOK_URL,
    blocks=(
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "✅ Task Succeeded"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": "*Run:*\n{{.Run.Name}}"},
                {"type": "mrkdwn", "text": "*Phase:*\n{{.Phase}}"},
            ],
        },
        {"type": "divider"},
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": "{{.Run.Project}}/{{.Run.Domain}}"},
            ],
        },
    ),
)

# Simple text notification on failure or timeout.
slack_failure = notify.Slack(
    on_phase=(ActionPhase.FAILED, ActionPhase.TIMED_OUT),
    webhook_url=SLACK_WEBHOOK_URL,
    message="🚨 Run {{.Run.Name}} failed with phase {{.Phase}}: {{.Error}}",
)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(
        env_vars={"SLACK_WEBHOOK_URL": SLACK_WEBHOOK_URL},
        notifications=(slack_success, slack_failure),
    ).run(compute, x=3, y=7)
    print(run.url)
