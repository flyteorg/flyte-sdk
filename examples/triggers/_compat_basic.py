"""Backward-compat test: regular (non-reusable) task with an inline-bound minutely trigger.

Uses minutely("start_time") so the kickoff arg is bound (inline-valid for non-offloading SDKs),
avoiding the bare hourly() pattern that the backend's inline validation rejects.
"""

import os
from datetime import datetime

import flyte

env = flyte.TaskEnvironment(name=os.environ.get("COMPAT_ENV", "compat_basic"), env_vars={"_U_USE_ACTIONS": "1"})


@env.task(triggers=flyte.Trigger.minutely("start_time"))
def compat_basic_task(start_time: datetime, x: int = 7) -> str:
    return f"compat_basic executed at {start_time.isoformat()} with x={x}"


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(env)
