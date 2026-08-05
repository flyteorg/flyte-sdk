"""
Exercises ``flyte.Timeout(max_runtime=...)`` — the per-attempt wall-clock
budget the leaseworker enforces against the time the plugin spent in
``PluginPhase=Running``.

The task is expected to reach Running quickly, then sleep for 10 minutes.
``max_runtime=30s`` should trip the leaseworker's check well before sleep ends.

Note: this is distinct from ``Timeout(deadline=...)``, which is enforced by
the leasor as an absolute budget across all attempts (and triggers
finalize/abort via the lease state machine, not the leaseworker's executor).
"""

import asyncio
from datetime import timedelta

import flyte

env = flyte.TaskEnvironment(name="max_runtime_demo", resources=flyte.Resources(cpu=1, memory="250Mi"))


@env.task(
    timeout=flyte.Timeout(max_runtime=timedelta(seconds=30)),
)
async def long_sleeper_runtime() -> str:
    print("long_sleeper_runtime: starting; will sleep 600s, max_runtime=30s")
    await asyncio.sleep(600)
    print("long_sleeper_runtime: WOKE UP — max_runtime did NOT fire (unexpected)")
    return "unexpectedly slept through max_runtime"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(long_sleeper_runtime)
    print(run.name)
    print(run.url)
