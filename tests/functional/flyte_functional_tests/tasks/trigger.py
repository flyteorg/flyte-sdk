"""Trigger (scheduled automation) check — deploy, verify the schedule spec, toggle.

No execution needed — the registered schedule is the signal.
"""

from __future__ import annotations

import os
from datetime import datetime

import flyte  # type: ignore

_suffix = os.environ.get("FLYTE_FUNCTIONAL_SUFFIX") or os.environ.get("ENV_SUFFIX") or "smoke"

_trigger_env = flyte.TaskEnvironment(name=f"functional-trigger-{_suffix}")


@_trigger_env.task(triggers=flyte.Trigger.daily(trigger_time_input_key="trigger_time"))
async def triggered_task(trigger_time: datetime, x: int = 1) -> str:
    return f"triggered at {trigger_time.isoformat()} x={x}"
