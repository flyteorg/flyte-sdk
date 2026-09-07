"""Triggers without automation: a named, pre-bound launch configuration.

A `Trigger` does not have to be scheduled. Leave `automation` unset and the
trigger becomes a saved launch configuration for the task -- default inputs,
queue, env vars, notifications -- that nothing fires on its own. It is fired
on demand only (from the UI, or via the API), which makes it a convenient way
to publish a handful of "blessed" ways to run a task without re-typing inputs.

Try it (see the env vars below for where notifications are delivered):

    SLACK_WEBHOOK_URL=... NOTIFICATION_EMAIL=... flyte deploy examples/triggers/manual.py env
    flyte get trigger

`report_on_demand` deploys with two such triggers, `quick-report` and
`full-report`, alongside a regular scheduled one. Fire either manual trigger
from the task's Triggers tab in the UI, or from Python with `flyte.run` (see
`programmatic.py`). The run starts with the trigger's bound inputs, env vars
and notifications.
"""

import os
from datetime import datetime

import flyte
import flyte.notify
from flyte.models import ActionPhase

env = flyte.TaskEnvironment(name="manual_trigger_example")

# Where notifications go. Read from the deploying shell so no credentials land
# in the example:
#
#     SLACK_WEBHOOK_URL=https://hooks.slack.com/services/... \
#     NOTIFICATION_EMAIL=you@example.com \
#     flyte deploy examples/triggers/manual.py env
#
# webhook.site is handy for seeing the Slack payload if you have no webhook yet.
SLACK_WEBHOOK = os.environ.get("SLACK_WEBHOOK_URL", "https://webhook.site/")
REPORT_RECIPIENTS = (os.environ.get("NOTIFICATION_EMAIL", "<EMAIL>"),)

# No `automation=`: nothing schedules these. Each is just a named set of
# inputs plus what to do when the run ends.
#
# Trigger inputs override the task's own defaults (`region="all"`, `days=7`
# below) for every run fired through the trigger. Inputs the trigger does not
# mention keep the task default, so `quick-report` still gets `as_of=None`.
quick_report = flyte.Trigger(
    name="quick-report",
    inputs={"region": "us-east", "days": 1},
    description="Yesterday only, for a fast sanity check",
    # A quick check only needs to shout when something goes wrong.
    notifications=flyte.notify.Slack(
        on_phase=(ActionPhase.FAILED, ActionPhase.TIMED_OUT),
        webhook_url=SLACK_WEBHOOK,
        message=":x: quick-report {{.Run.Name}} ended in {{.Phase}}: {{.Error}}",
    ),
)

full_report = flyte.Trigger(
    name="full-report",
    inputs={"region": "all", "days": 30},
    description="The full monthly report",
    env_vars={"REPORT_VERBOSE": "1"},
    # The monthly report is worth an email on success and a Slack ping on failure.
    notifications=(
        flyte.notify.Email(
            on_phase=ActionPhase.SUCCEEDED,
            recipients=REPORT_RECIPIENTS,
            subject="Monthly report {{.Run.Name}} is ready",
            body="The full report finished.\nRun: {{.Run.Name}}\nProject/Domain: {{.Run.Project}}/{{.Run.Domain}}",
        ),
        flyte.notify.Slack(
            on_phase=ActionPhase.FAILED,
            webhook_url=SLACK_WEBHOOK,
            message=":rotating_light: full-report {{.Run.Name}} failed: {{.Error}}",
        ),
    ),
)

# A scheduled trigger can sit next to the manual ones on the same task. Only a
# schedule can bind `flyte.TriggerTime`, since a manual trigger has no fire time.
nightly = flyte.Trigger(
    name="nightly",
    automation=flyte.Cron("0 2 * * *"),
    inputs={"as_of": flyte.TriggerTime, "region": "all", "days": 1},
    notifications=flyte.notify.Slack(
        on_phase=ActionPhase.FAILED,
        webhook_url=SLACK_WEBHOOK,
        message=":rotating_light: nightly report {{.Run.Name}} failed: {{.Error}}",
    ),
)


@env.task(triggers=(quick_report, full_report, nightly))
async def report_on_demand(region: str = "all", days: int = 7, as_of: datetime | None = None) -> str:
    as_of = as_of or datetime.now()
    msg = f"report for region={region!r} over the last {days} day(s), as of {as_of.isoformat()}"
    print(msg)
    return msg


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(env)
