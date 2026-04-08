"""
Map Concurrency Verification
=============================
Demonstrates the concurrency bug in MapAsyncIterator._initialize():

  Expected (after fix): with concurrency=2 and 8 tasks, tasks start in waves of 2.
  Bug (current):        all 8 tasks start immediately regardless of concurrency.

How to read the output
-----------------------
Each `timed_task` returns its Unix start time.  The parent task sorts them and
prints the offset from the earliest start.

If concurrency=2 is respected you see two groups of start times:

    task 0  +0.0 s  ─┐
    task 1  +0.1 s  ─┘ wave 1
    task 2  +5.0 s  ─┐
    task 3  +5.1 s  ─┘ wave 2 (after wave 1 finishes)
    …

If the bug is present all tasks show +0.x s.

Run
----
    python examples/basics/map_concurrency_verify.py
"""

import time
from typing import List

import flyte

env = flyte.TaskEnvironment(name="map-concurrency-verify")

_TASK_DURATION_S = 5  # how long each task "works"


@env.task
async def timed_task(x: int) -> float:
    """Sleep to hold a concurrency slot, then return the Unix timestamp of when we started."""
    import asyncio

    start = time.time()
    print(f"task {x} started")
    await asyncio.sleep(_TASK_DURATION_S)
    return start


@env.task
async def verify_map_concurrency(n: int, concurrency: int) -> List[str]:
    """
    Run `timed_task` n times with the given concurrency limit and print a report.

    Look at the 'started at' offsets in the output:
    - Bug present  → all offsets ≈ 0 s  (all tasks launched at once)
    - Bug fixed    → offsets increase in steps of ~5s per wave
    """
    start_times: List[float] = []
    async for result in flyte.map.aio(
        timed_task,
        range(n),
        concurrency=concurrency,
        return_exceptions=True,
    ):
        if isinstance(result, Exception):
            raise result
        start_times.append(result)

    t0 = min(start_times)
    lines: List[str] = []
    for i, t in enumerate(sorted(start_times)):
        lines.append(f"task {i:2d}  started at +{t - t0:.1f}s")

    # count tasks that started within the first wave window
    first_wave = sum(1 for t in start_times if t - t0 < _TASK_DURATION_S * 0.5)

    print(f"\n=== Map concurrency={concurrency}, n={n} ===")
    for line in lines:
        print(line)
    print(f"\nTasks that started in first wave: {first_wave}  (expected ≤ {concurrency})")

    if first_wave <= concurrency:
        print("PASS: concurrency limit appears to be respected.")
    else:
        print(f"BUG: {first_wave} tasks started at once, expected at most {concurrency}.")

    return lines


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(verify_map_concurrency, 8, 2)
    print(run.url)
