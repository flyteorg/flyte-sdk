"""Wide fan-out: a very large action count under a tight cap.

Runs locally: every child is a core-sleep leaf (the Sleep plugin), so it executes
in leaseworker and creates no task pods. That lets us fan out wide without
consuming real k8s resources.

Stresses the scheduler-side trimming (capActionsPerRun runs every 10ms over a big
batch) and the per-run byRun scan in the 1s snapshot loop. The cap must still hold
exactly while these paths do real work each tick.

Run:   python examples/concurrency/wide_fanout.py
Watch: /debug poll loop (LIMIT=3); leasor schedule + snapshot timers
       (ObserveScheduleDuration, ObserveQueueSnapshotApply) should stay flat.
Pass:  per-run active never exceeds 3; timers don't blow up with N;
       run_concurrency_tracked_runs -> 0.
"""

import asyncio
from datetime import timedelta

import flyte
from flyte.extras import Sleep

# Leaves run in leaseworker via the core-sleep plugin: no task pods are created,
# so we can fan out wide without paying pod-startup cost.
sleep_env = flyte.TaskEnvironment(name="conc_wide_fanout_leaf", plugin_config=Sleep())

env = flyte.TaskEnvironment(name="conc_wide_fanout", depends_on=[sleep_env])

# --- knobs ---
N = 2000
CAP = 3
SLEEP_S = 1.0


@sleep_env.task
async def step(duration: timedelta) -> None:
    return None


@env.task
async def main(n: int = N) -> int:
    tasks = [step(duration=timedelta(seconds=SLEEP_S)) for _ in range(n)]
    await asyncio.gather(*tasks)
    # Return a scalar, not the N-element list, to avoid a huge run-output blob.
    return n


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(max_action_concurrency=CAP).run(main, n=N)
    print(run.url)
    run.wait()
