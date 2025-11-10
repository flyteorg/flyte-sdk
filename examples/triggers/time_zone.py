# examples/triggers/hourly.py

from datetime import datetime

import flyte

env = flyte.TaskEnvironment(
    name="time_zone_trigger",
)

nyc_trigger = flyte.Trigger(
    "nyc_tz",
    flyte.Cron(
        "1 12 * * *", timezone="America/New_York"
    ), # Every day at 12:01 PM ET
    inputs={"start_time": flyte.TriggerTime, "x": 1},
)

sf_trigger = flyte.Trigger(
    "sf_tz",
    flyte.Cron(
        "0 9 * * *", timezone="America/Los_Angeles"
    ), # Every day at 9 AM PT
    inputs={"start_time": flyte.TriggerTime, "x": 1},
)


@env.task(triggers=(nyc_trigger, sf_trigger))
def my_task(start_time: datetime, x: int) -> str:
    return f"Task executed at {start_time.isoformat()} with x={x}"


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(env)
