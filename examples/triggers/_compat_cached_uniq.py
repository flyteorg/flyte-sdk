"""Decisive cache-hit experiment: a uniquely-named cached task (never ad-hoc run) on a minutely
trigger with start_time ignored. If caching works: fire 1 misses+runs, later fires hit cache and
freeze start_time. Distinguishes 'cache hit' from 'CreateRun failure'.
"""

from datetime import datetime

import flyte

env = flyte.TaskEnvironment(name="compat_cache_exp", env_vars={"_U_USE_ACTIONS": "1"})


@env.task(
    cache=flyte.Cache(behavior="auto", ignored_inputs="start_time"),
    triggers=flyte.Trigger.minutely("start_time", name="minutely_cached"),
)
def cache_exp_unique_task(start_time: datetime, x: int = 1) -> str:
    return f"cache_exp executed at {start_time.isoformat()} with x={x}"


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(env)
