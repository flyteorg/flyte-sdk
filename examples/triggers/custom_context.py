"""
Example of using custom_context with Triggers to propagate metadata to scheduled runs.

custom_context on a Trigger works the same way as custom_context on
flyte.with_runcontext() — tasks read it via flyte.ctx().custom_context.
"""

from datetime import datetime

import flyte

env = flyte.TaskEnvironment(
    name="trigger_with_custom_context",
    image=flyte.Image.from_debian_base(),
)


# Trigger using a convenience method with custom_context
minutely_monitor = flyte.Trigger.minutely(
    trigger_time_input_key="trigger_time",
    name="minutely_monitor",
    custom_context={"team": "infra", "alert_channel": "#ops"},
)


@env.task(triggers=minutely_monitor)
async def monitor_task(trigger_time: datetime) -> str:
    ctx = flyte.ctx().custom_context
    return f"Monitor ran for team={ctx.get('team')}"


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(env)
