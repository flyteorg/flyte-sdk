"""
Strict serialization with a concurrency-1 queue.

The `serial-1` queue is configured with `maxActionConcurrency=1`, which means
at most ONE task pinned to it is ever in flight, no matter how many you
submit. The remaining tasks wait in the queue and dispatch one at a time as
each predecessor completes.

Run it (against a cluster where `serial-1` exists):

    flyte run examples/queues/serial_one.py main

Expected timing for `count=5, sleep_seconds=4`: total wall time ~= 5 * 4s ~= 20s,
because the tasks are dispatched strictly one after another.

Use this to confirm that a 1-at-a-time queue actually serializes work even
when many tasks are submitted in a burst.
"""

import asyncio
from datetime import datetime, timezone
from functools import partial

import flyte

env = flyte.TaskEnvironment(
    name="queues_serial_one",
    resources=flyte.Resources(memory="200Mi"),
)


# Child task pinned to the serial-1 queue (MaxActionConcurrency=1). The
# parent `main` below does NOT override the queue, so it uses the default
# routing and avoids any cluster-pool validation surprises at run creation.
@env.task(queue="serial-1")
async def step(i: int, sleep_seconds: int) -> str:
    """One unit of serialized work. Prints start/end so you can read the log
    timestamps and confirm no two steps overlap."""
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[serial-1] step {i} START at {started}", flush=True)
    await asyncio.sleep(sleep_seconds)
    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[serial-1] step {i} END   at {finished}", flush=True)
    return f"step {i} done"


@env.task
async def main(count: int = 5, sleep_seconds: int = 4) -> list[str]:
    """Fan out `count` steps in parallel. Without queue serialization they'd
    all start at once; with `maxActionConcurrency=1` they execute in series."""
    return list(flyte.map(partial(step, sleep_seconds=sleep_seconds), list(range(count))))


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, count=5, sleep_seconds=4)
    print(run.name)
    print(run.url)
    run.wait()
