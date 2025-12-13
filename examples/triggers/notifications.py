from datetime import datetime

import flyte
import flyte.notify

env = flyte.TaskEnvironment(
    name="example_task",
)

trig1 = flyte.Trigger(
    name="hourly",
    auto_activate=True,
    automation=flyte.Cron("* * * * *"),
    notifications=flyte.notify.Slack(
        on_phase="FAILED",
        webhook_url="https://webhook.site/",
        message="Hello world! from {task.name}",
        blocks=(
            {"type": "header", "text": {"type": "plain_text", "text": "🚨 Run Failed"}},
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": "*Workflow:*\n{run.name}"},
                    {"type": "mrkdwn", "text": "*Error:*\n{run.error}"},
                ],
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View Logs"},
                        "url": "https://logs.example.com/{run.name}",
                    }
                ],
            },
        ),
    ),
)

trig2 = flyte.Trigger(
    name="hourly",
    auto_activate=True,
    automation=flyte.Cron("* * * * *"),
    notifications=(
        flyte.notify.Slack(
            on_phase=(
                "FAILED",
                "TIMED OUT",
            ),
            webhook_url="https://webhook.site/",
            message="Hello world! from {task.name}",
        ),
        flyte.notify.Webhook(
            on_phase="SUCCEEDED",
            headers={"Content-Type": "application/json"},
            url="https://webhook.site/",
            body={
                "xyz": "{run.name}",
            },
        ),
        flyte.notify.Email(
            on_phase="SUCCEEDED",
            subject="Hello world! from {task.name}",
            body="Hello world! from {task.name}",
            recipients=("<EMAIL>",),
        ),
    ),
)


@env.task(triggers=(trig1, trig2))  # Every hour
def example_task(trigger_time: datetime, x: int = 1) -> str:
    return f"Task executed at {trigger_time.isoformat()} with x={x}"
