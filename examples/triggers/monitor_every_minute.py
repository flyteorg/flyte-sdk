"""
Minimal trigger example.

Every minute, the trigger fires `parent_monitor`, which invokes `child_probe`.
The child sleeps for 1 second and returns. The parent returns a short summary.

CLI cheat sheet.

Trigger name: "minutely".
Server-side task name is "<env_name>.<function_name>", i.e.
    minutely_monitor_example.parent_monitor

    # Deploy the env (registers tasks + trigger, auto-activates it)
    python examples/triggers/monitor_every_minute.py

    # List triggers (optionally scoped to one task)
    flyte get trigger
    flyte get trigger minutely_monitor_example.parent_monitor

    # Show one trigger's details
    flyte get trigger minutely_monitor_example.parent_monitor minutely

    # Pause / resume without redeploying
    flyte update trigger minutely minutely_monitor_example.parent_monitor --deactivate
    flyte update trigger minutely minutely_monitor_example.parent_monitor --activate

    # Remove the trigger entirely
    flyte delete trigger minutely minutely_monitor_example.parent_monitor

All commands accept --project <p> --domain <d> to target a non-default project/domain.
"""

import asyncio
from datetime import datetime

import flyte

env = flyte.TaskEnvironment(
    name="minutely_monitor_example",
)


@env.task
async def child_probe(trigger_time: datetime) -> str:
    await asyncio.sleep(1)
    return f"probe ok @ {trigger_time.isoformat()}"


@env.task(triggers=flyte.Trigger.minutely("trigger_time"))
async def parent_monitor(trigger_time: datetime) -> str:
    result = await child_probe(trigger_time=trigger_time)
    return f"parent saw: {result}"


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(env)
