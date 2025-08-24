"""
The Flyte Trigger module provides functionality to create and manage triggers that can be associated with Flyte tasks.
Triggers can be used to run tasks on a schedule, in response to events, or based on other conditions.

Example:
```python
from datetime import datetime
import flyte
import flyte.trigger

env = flyte.TaskEnvironment(...)

@env.task(trigger=flyte.trigger.daily())
def example_task(trigger_time: datetime, x: int = 1) -> str:
    return f"Task executed at {trigger_time.isoformat()} with x={x}"

custom_trigger = flyte.trigger.new(
    "custom_cron", flyte.trigger.Cron("0 0 * * *"),
     inputs={"start_time": flyte.trigger.TriggerTime, "x": 1},
 )

@env.task(trigger=custom_trigger)
def custom_task(start_time: datetime, x: int = 11) -> str:
    return f"Custom task executed at {start_time.isoformat()} with x={x}"

@env.task(trigger=(flyte.trigger.every_minute(), flyte.trigger.hourly()))
def multi_trigger_task(trigger_time: datetime, x: int = 1) -> str:
    return f"Multi-trigger task executed at {trigger_time.isoformat()} with x={x
```

"""

from ._defs import TriggerTime
from ._globals import daily, every_minute, hourly, weekly
from ._schedule import Cron, cron_daily, cron_hourly, cron_minute, cron_weekly
from ._trigger import Trigger, new

__all__ = [
    "Cron",
    "Trigger",
    "TriggerTime",
    "cron_daily",
    "cron_hourly",
    "cron_minute",
    "cron_weekly",
    "daily",
    "every_minute",
    "hourly",
    "new",
    "weekly",
]
