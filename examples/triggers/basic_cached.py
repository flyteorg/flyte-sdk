"""
Example of a cached task on an every-minute trigger, with the trigger time excluded from the
cache key.

Each minute the `minutely_cached` trigger fires with a different `start_time`, but because it is
listed in `ignored_inputs` the cache key stays constant (and `x` defaults to 1). So:

- the first fire is a cache miss and runs the task, and
- subsequent fires hit the cache and return the *first* run's `start_time`.

If you watch the run outputs minute over minute and the timestamp stays frozen at the first fire,
caching + start_time-exclusion works as intended.
"""

from datetime import datetime

import flyte

env = flyte.TaskEnvironment(
    name="cached_example_task",
    env_vars={"_U_USE_ACTIONS": "1"},
)


@env.task(
    cache=flyte.Cache(behavior="auto", ignored_inputs="start_time"),
    triggers=flyte.Trigger.minutely("start_time", name="minutely_cached"),
)
def cached_task(start_time: datetime, x: int = 1) -> str:
    return f"Cached task executed at {start_time.isoformat()} with x={x}"


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(env)
