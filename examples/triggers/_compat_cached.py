"""Backward-compat test: cached task on a minutely trigger with the trigger time excluded from
the cache key (ignored_inputs="start_time"). Inline-bound minutely so it is valid for non-offloading
SDKs. Expected: first fire misses+runs, later fires hit cache and return the FROZEN first start_time.

COMPAT_ENV selects the env name so multiple SDK versions can coexist.
"""

import os
from datetime import datetime

import flyte

env = flyte.TaskEnvironment(name=os.environ.get("COMPAT_ENV", "compat_cached"), env_vars={"_U_USE_ACTIONS": "1"})


@env.task(
    cache=flyte.Cache(behavior="auto", ignored_inputs="start_time"),
    triggers=flyte.Trigger.minutely("start_time", name="minutely_cached"),
)
def compat_cached_task(start_time: datetime, x: int = 1) -> str:
    return f"compat_cached executed at {start_time.isoformat()} with x={x}"


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(env)
