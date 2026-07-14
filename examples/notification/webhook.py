"""Custom HTTP webhook notification example.

The Webhook notifier is the most flexible option: you control the URL, HTTP
method, headers, and JSON body. All string values (including header values and
nested body values) support template variables.

This is handy for integrating with PagerDuty, Opsgenie, a custom alerting
service, or a test endpoint like https://webhook.site.

Usage:
    export ALERT_WEBHOOK_URL="https://webhook.site/your-unique-id"
    export ALERT_API_KEY="secret"
    python examples/notification/webhook.py

Template variables available in url/headers/body:
    {{.Run.Name}}, {{.Run.Project}}, {{.Run.Domain}}, {{.Phase}}, {{.Error}}
"""

import os

import flyte
from flyte import notify
from flyte.models import ActionPhase
from flyte.remote import Run

env = flyte.TaskEnvironment(
    name="notify_webhook",
    resources=flyte.Resources(memory="250Mi"),
)

ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "https://webhook.site/replace-me")
ALERT_API_KEY = os.environ.get("ALERT_API_KEY", "secret")


@env.task
def compute(x: int, y: int) -> int:
    return x + y


# POST a structured JSON payload on any terminal phase.
webhook_any = notify.Webhook(
    on_phase=(ActionPhase.SUCCEEDED, ActionPhase.FAILED, ActionPhase.ABORTED, ActionPhase.TIMED_OUT),
    url=ALERT_WEBHOOK_URL,
    method="POST",
    headers={
        "Content-Type": "application/json",
        "X-API-Key": ALERT_API_KEY,
        "X-Run-Name": "{{.Run.Name}}",
    },
    body={
        "run": "{{.Run.Name}}",
        "project": "{{.Run.Project}}",
        "domain": "{{.Run.Domain}}",
        "phase": "{{.Phase}}",
        "error": "{{.Error}}",
    },
)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(
        env_vars={"ALERT_WEBHOOK_URL": ALERT_WEBHOOK_URL, "ALERT_API_KEY": ALERT_API_KEY},
        notifications=webhook_any,
    ).run(compute, x=3, y=7)
    assert isinstance(run, Run)

    print(run.name)
    print(run.url)
    run.wait()
