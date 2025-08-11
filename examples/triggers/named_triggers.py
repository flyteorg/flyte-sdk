"""
Examples of using named triggers in Flyte.

This example demonstrates how to create and use named triggers in Flyte. Multiple triggers are defined, including
cron-based triggers and a webhook trigger. The `example_task` function is decorated with these triggers, allowing it to
be executed based on the specified schedules or webhook events.
"""

import flyte
import flyte.trigger

env = flyte.TaskEnvironment(
    "named_triggers",
)

daily_trigger = flyte.trigger.Cron("daily", "0 * * * *")
weekly_trigger = flyte.trigger.Cron("weekly", "0 0 * * 0", auto_activate=False)  # Every Sunday at midnight
monthly_trigger = flyte.trigger.Cron("monthly", "0 0 1 * *", {"x": 10})  # First day of every month with x=10
webhook_trigger = flyte.trigger.Webhook(
    "web1", {"start_time": "2023-01-01T00:00:00Z", "x": 5}
)  # Example webhook trigger with parameters


@env.task(trigger=(daily_trigger, weekly_trigger, monthly_trigger, webhook_trigger))
def example_task(start_time: str, x: int = 1) -> str:
    return f"{start_time} {x}"


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    flyte.deploy(env)
