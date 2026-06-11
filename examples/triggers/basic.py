"""
Example of a simple task that runs on a schedule using Flyte's Cron trigger.

"""

from datetime import datetime

import flyte

env = flyte.TaskEnvironment(
    name="example_task",
)


@env.task(triggers=flyte.Trigger.hourly())  # Every hour
def example_task(trigger_time: datetime, x: int = 1) -> str:
    return f"Task executed at {trigger_time.isoformat()} with x={x}"


custom_trigger = flyte.Trigger(
    "custom_cron",
    flyte.Cron("0 0 * * *", timezone="Europe/London"),  # Runs once every day at midnight London time
    inputs={"start_time": flyte.TriggerTime, "x": 1},
)


@env.task(triggers=(custom_trigger, flyte.Trigger.minutely("start_time")))  # Custom trigger and every minute
def custom_task(start_time: datetime, x: int = 11) -> str:
    return f"Custom task executed at {start_time.isoformat()} with x={x}"


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(env)
