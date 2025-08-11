"""
Example of a simple task that runs on a schedule using Flyte's Cron trigger.

"""

from datetime import datetime

import flyte
import flyte.trigger

env = flyte.TaskEnvironment(
    name="example_task",
)


# If no input bound is specified, flyte will automatically bind the trigger time to the first argument with
# time datetime
# In case of ambiguous "datetime" or not enough defaults, an error will be raised.
@env.task(trigger=flyte.trigger.Cron("0 * * * *"))  # Every hour
def example_task(start_time: datetime, x: int = 1) -> str:
    return f"Task executed at {start_time.isoformat()} with x={x}"


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    flyte.deploy(env)
