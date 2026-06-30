"""
Two queues, independent caps.

Two task environments target two different queues:

  - `bulk-a` with `maxActionConcurrency=2`
  - `bulk-b` with `maxActionConcurrency=4`

Tasks on one queue do not consume capacity on the other. When this workflow
fans out the same number of tasks into both queues at once, `bulk-b` should
finish noticeably faster than `bulk-a` purely because its cap is higher —
the work on either queue is otherwise identical.

Run it:

    flyte run examples/queues/multi_queue.py main

Expected timing for `count_each=8, sleep_seconds=3`:

  - bulk-a (cap 2): ceil(8/2) * 3s = 12s
  - bulk-b (cap 4): ceil(8/4) * 3s = 6s

Both run in parallel, so the workflow as a whole finishes at ~12s. Read the
step START/END timestamps to confirm:

  - on `bulk-a`, at most two `[bulk-a]` lines overlap at any time;
  - on `bulk-b`, at most four `[bulk-b]` lines overlap;
  - `[bulk-a]` and `[bulk-b]` lines from the same wallclock interval freely
    overlap with each other — the queues do not contend.
"""

import asyncio
import logging
from datetime import datetime
from functools import partial

import flyte

env_a = flyte.TaskEnvironment(
    name="queues_multi_a",
    resources=flyte.Resources(memory="200Mi"),
)

env_b = flyte.TaskEnvironment(
    name="queues_multi_b",
    resources=flyte.Resources(memory="200Mi"),
    depends_on=[env_a],
)


# Per-step queue overrides. The parent `main` uses default routing so run
# creation doesn't trip on cluster validation.
@env_a.task(queue="bulk-a")
async def step_a(i: int, sleep_seconds: int) -> str:
    started = datetime.utcnow().isoformat(timespec="seconds")
    print(f"[bulk-a] step {i} START at {started}", flush=True)
    await asyncio.sleep(sleep_seconds)
    finished = datetime.utcnow().isoformat(timespec="seconds")
    print(f"[bulk-a] step {i} END   at {finished}", flush=True)
    return f"a:{i}"


@env_b.task(queue="bulk-b")
async def step_b(i: int, sleep_seconds: int) -> str:
    started = datetime.utcnow().isoformat(timespec="seconds")
    print(f"[bulk-b] step {i} START at {started}", flush=True)
    await asyncio.sleep(sleep_seconds)
    finished = datetime.utcnow().isoformat(timespec="seconds")
    print(f"[bulk-b] step {i} END   at {finished}", flush=True)
    return f"b:{i}"


# Driver uses default routing (no per-task queue override) so it stays out
# of either bulk-* cap and run creation routes through the default cluster.
@env_b.task
async def main(count_each: int = 8, sleep_seconds: int = 3) -> dict[str, list[str]]:
    a_results = list(flyte.map(partial(step_a, sleep_seconds=sleep_seconds), list(range(count_each))))
    b_results = list(flyte.map(partial(step_b, sleep_seconds=sleep_seconds), list(range(count_each))))
    return {"a": a_results, "b": b_results}


if __name__ == "__main__":
    flyte.init_from_config(log_level=logging.DEBUG)
    run = flyte.run(main, count_each=8, sleep_seconds=3)
    print(run.name)
    print(run.url)
    run.wait()
