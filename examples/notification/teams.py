"""Microsoft Teams notification example.

Sends a Teams message when a run reaches a terminal phase. A simple title +
message notification is shown for success, and a richer Adaptive Card is shown
for failures (when a card is provided, title and message are ignored).

Usage:
    export TEAMS_WEBHOOK_URL="https://outlook.office.com/webhook/YOUR/WEBHOOK/URL"
    python examples/notification/teams.py

Template variables available in title/message/card:
    {{.Run.Name}}, {{.Run.Project}}, {{.Run.Domain}}, {{.Phase}}, {{.Error}}
"""

import os

import flyte
from flyte import notify
from flyte.models import ActionPhase
from flyte.remote import Run

env = flyte.TaskEnvironment(
    name="notify_teams",
    resources=flyte.Resources(memory="250Mi"),
)

TEAMS_WEBHOOK_URL = os.environ.get("TEAMS_WEBHOOK_URL", "https://outlook.office.com/webhook/REPLACE/ME")


@env.task
def compute(x: int, y: int) -> int:
    return x + y


# Simple title + message on success.
teams_success = notify.Teams(
    on_phase=ActionPhase.SUCCEEDED,
    webhook_url=TEAMS_WEBHOOK_URL,
    title="✅ Run {{.Run.Name}} succeeded",
    message="Run {{.Run.Name}} finished with phase {{.Phase}} in {{.Run.Project}}/{{.Run.Domain}}",
)

# Rich Adaptive Card on failure.
teams_failure = notify.Teams(
    on_phase=(ActionPhase.FAILED, ActionPhase.TIMED_OUT),
    webhook_url=TEAMS_WEBHOOK_URL,
    card={
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": [
            {
                "type": "TextBlock",
                "size": "Large",
                "weight": "Bolder",
                "color": "Attention",
                "text": "🚨 Run Failed",
            },
            {
                "type": "FactSet",
                "facts": [
                    {"title": "Run", "value": "{{.Run.Name}}"},
                    {"title": "Phase", "value": "{{.Phase}}"},
                    {"title": "Project/Domain", "value": "{{.Run.Project}}/{{.Run.Domain}}"},
                    {"title": "Error", "value": "{{.Error}}"},
                ],
            },
        ],
    },
)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(
        env_vars={"TEAMS_WEBHOOK_URL": TEAMS_WEBHOOK_URL},
        notifications=(teams_success, teams_failure),
    ).run(compute, x=3, y=7)
    assert isinstance(run, Run)

    print(run.name)
    print(run.url)
    run.wait()
