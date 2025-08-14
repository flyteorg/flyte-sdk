"""
Example of a simple task that runs on a schedule using Flyte's Cron trigger.

"""

from datetime import datetime

import flyte
import flyte.trigger

env = flyte.TaskEnvironment(
    name="example_task",
)


@env.task(trigger=flyte.trigger.hourly())  # Every hour
def example_task(trigger_time: datetime, x: int = 1) -> str:
    return f"Task executed at {trigger_time.isoformat()} with x={x}"


custom_trigger = flyte.trigger.new(
    "custom_cron",
    flyte.trigger.Cron("0 0 * * *"),  # Runs every day
    inputs={"start_time": flyte.trigger.TriggerTime, "x": 1},
)


@env.task(trigger=(custom_trigger, flyte.trigger.every_minute("start_time")))  # Custom trigger and every minute
def custom_task(start_time: datetime, x: int = 11) -> str:
    return f"Custom task executed at {start_time.isoformat()} with x={x}"


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    flyte.deploy(env)
