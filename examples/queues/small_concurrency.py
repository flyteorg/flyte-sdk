"""
Bounded parallelism with a small-N queue.

The `small-3` queue is configured with `maxActionConcurrency=3`. Submitting
more than 3 tasks at once does not violate the cap — the queue holds the
overflow and dispatches additional work only as in-flight tasks complete.

Run it:

    flyte run examples/queues/small_concurrency.py main

Expected timing for `count=10, sleep_seconds=4` and concurrency=3:
ceil(10/3) * 4s = 16s wall time. If you see <12s, the cap is not being
enforced; if you see >>20s, the queue is under-scheduling.

Use this to validate that the queue cap holds under burst load and that
fresh capacity is recycled promptly as each task finishes.
"""

import asyncio
from datetime import datetime, timezone
from functools import partial

import flyte

env = flyte.TaskEnvironment(
    name="queues_small_concurrency",
    resources=flyte.Resources(memory="200Mi"),
)


# Child task pinned to small-3 (MaxActionConcurrency=3). The parent `main`
# uses default routing so run creation doesn't trip on cluster validation.
@env.task(queue="small-3")
async def step(i: int, sleep_seconds: int) -> str:
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[small-3] step {i} START at {started}", flush=True)
    await asyncio.sleep(sleep_seconds)
    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[small-3] step {i} END   at {finished}", flush=True)
    return f"step {i} done"


@env.task
async def main(count: int = 10, sleep_seconds: int = 4) -> list[str]:
    return list(flyte.map(partial(step, sleep_seconds=sleep_seconds), list(range(count))))


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, count=10, sleep_seconds=4)
    print(run.name)
    print(run.url)
    run.wait()
