"""Nested fan-out — characterizes the known deadlock at cap <= nesting depth.

Two parent levels: main -> step -> leaf. Each parent holds a run slot while it
awaits its children. So with a small cap, the awaiting parents consume every slot
and their children can never be scheduled — the run wedges. This is the same
mechanism that makes cap=1 a guaranteed deadlock, one level deeper.

Expected:
  CAP = 2  -> WEDGES. a0(main) + 1 step fill both slots; leaves never schedule.
             Run sits at active=2, schedule_skip_total{run_at_action_concurrency}
             climbs, nothing progresses. Abort it.
  CAP = 3  -> COMPLETES (serially): a0 + 1 step + 1 leaf fits, so one leaf runs
             at a time and the run drains.

This is a "characterize the sharp edge" test, not a pass/fail bug — it documents
that the limit must exceed the run's concurrently-awaiting-parent depth.

Run:   python examples/concurrency/nested_fanout_deadlock.py   # CAP=2, expect wedge
Watch: /debug poll loop; if active pins at CAP with zero progress for minutes,
       you've reproduced the deadlock. Abort to clean up.
"""

import asyncio

import flyte

env = flyte.TaskEnvironment(name="conc_nested_deadlock")

# --- knobs ---
N = 4  # number of step (mid-level parent) actions
CHILDREN = 3  # leaves per step
CAP = 2  # try 2 (wedges) vs 3 (completes)
SLEEP_S = 3


@env.task
async def leaf(j: int) -> int:
    await asyncio.sleep(SLEEP_S)
    return j


@env.task
async def step(i: int) -> int:
    # step is itself a parent: it holds a slot while awaiting its leaves.
    tasks = [leaf(j) for j in range(CHILDREN)]
    results = await asyncio.gather(*tasks)
    return sum(results)


@env.task
async def main(n: int = N) -> list[int]:
    tasks = [step(i) for i in range(n)]
    results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(max_action_concurrency=CAP).run(main, n=N)
    print("run name (abort handle if it wedges):", run.name)
    print(run.url)
    run.wait()
