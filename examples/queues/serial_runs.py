"""
Serialize runs, not actions.

The `runs-1` queue is configured with `maxRunConcurrency=1` and no
`maxActionConcurrency` cap. That means:

  - At most ONE run (workflow execution) targets this queue at a time.
  - Inside the one active run, the workflow's child actions can fan out
    freely — hundreds of children per run is fine, all running in parallel
    (subject to whatever capacity your workers have).

Use this when you have a job that internally parallelizes well but must
not start until any prior invocation of the same job has finished — e.g.
a training run that updates a shared checkpoint, or a batch job that
must not overlap with itself.

How it works:

  - `main` is decorated with `@env.task(queue="runs-1")`. The SDK sets
    RunSpec.cluster to `runs-1`, so workflow-service routes the run there
    and the root action gets pinned to the queue. The queue counts only
    root actions (a0) against `maxRunConcurrency`, so this is what gates
    overlapping invocations.
  - `child` is decorated with plain `@env.task` — no queue override, so
    children run on the default routing and never count against the
    run-cap.

Run it (submits NUM_RUNS runs concurrently from one script):

    python examples/queues/serial_runs.py             # 3 runs, default
    NUM_RUNS=5 python examples/queues/serial_runs.py  # 5 runs

You can also submit one run via the standard flyte run CLI:

    flyte run examples/queues/serial_runs.py main

Expected behavior with NUM_RUNS=3: all 3 runs are submitted within a few
hundred milliseconds, but the `[run-N] main START` timestamps land
strictly in series — the second run's START is after the first run's END,
the third after the second. Inside one run, the 50 children all start
within a tick of each other.
"""

import asyncio
import os
from datetime import datetime, timezone
from functools import partial

import flyte

env = flyte.TaskEnvironment(
    name="queues_serial_runs",
    resources=flyte.Resources(memory="200Mi"),
)


@env.task
async def child(i: int, sleep_seconds: int) -> str:
    """Unrestricted child task. Hundreds can be in flight at the same
    time — `runs-1` does not cap individual actions, only roots."""
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"child {i} START at {started}", flush=True)
    await asyncio.sleep(sleep_seconds)
    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"child {i} END   at {finished}", flush=True)
    return f"child{i}"


@env.task(queue="runs-1")
async def main(fan_out: int = 50, child_sleep_seconds: int = 2) -> list[str]:
    """Root action pinned to runs-1. Only one of these can be in `Sent`
    state at a time, regardless of how many times you submit it in
    parallel — the rest queue up and dispatch one at a time."""
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"main START at {started} fan_out={fan_out}", flush=True)
    results = list(flyte.map(partial(child, sleep_seconds=child_sleep_seconds), list(range(fan_out))))
    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"main END   at {finished} ({len(results)} children)", flush=True)
    return results


async def submit_concurrent_runs(num_runs: int, fan_out: int, child_sleep_seconds: int):
    """Submit `num_runs` runs of `main` in parallel via flyte.run.aio.
    Each call returns once the run is created (not when it completes), so
    all submissions land within a few hundred ms. Leasor then dispatches
    the roots one at a time because runs-1 caps maxRunConcurrency to 1."""
    print(f"submitting {num_runs} concurrent runs of main(fan_out={fan_out})")
    runs = await asyncio.gather(
        *[flyte.run.aio(main, fan_out=fan_out, child_sleep_seconds=child_sleep_seconds) for _ in range(num_runs)]
    )
    for i, r in enumerate(runs):
        print(f"  [{i}] submitted: {r.name}\n      {r.url}")
    print("waiting for all runs to reach terminal state...")
    await asyncio.gather(*[r.wait.aio() for r in runs])
    print("all runs done")


if __name__ == "__main__":
    flyte.init_from_config()
    num_runs = int(os.environ.get("NUM_RUNS", "3"))
    fan_out = int(os.environ.get("FAN_OUT", "50"))
    sleep_s = int(os.environ.get("CHILD_SLEEP", "2"))
    asyncio.run(submit_concurrent_runs(num_runs, fan_out, sleep_s))
