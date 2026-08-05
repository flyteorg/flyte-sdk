"""
Queue depth backpressure.

The `depth-limited` queue is configured with `maxDepth=5` — it can hold at
most 5 in-flight + waiting tasks at any time. When the queue is full, the
SDK's submission call gets `RESOURCE_EXHAUSTED` immediately. This is
backpressure to your caller, not a silent retry: it is the right place to
slow down a producer.

Run it:

    flyte run examples/queues/depth_backpressure.py main

For `count=8` against `maxDepth=5`, you should see ~5 tasks admitted and
~3 fail-fast with a backpressure error. Reduce `count` below 5 to see all
tasks accepted; raise it to see the rejection rate scale.

The number of rejections is timing-dependent — if one task happens to
complete (and free its slot) before the next submission attempts to admit,
that next task will succeed instead of being rejected. The point is that
total in-flight never exceeds 5, not that exactly 3 will always fail.
"""

import asyncio
from datetime import datetime, timezone
from functools import partial

import flyte

env = flyte.TaskEnvironment(
    name="queues_depth_backpressure",
    resources=flyte.Resources(memory="200Mi"),
)


# Child task pinned to depth-limited (MaxDepth=5). The parent `main` uses
# default routing so run creation doesn't trip on cluster validation.
@env.task(queue="depth-limited")
async def step(i: int, sleep_seconds: int) -> str:
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[depth-limited] step {i} START at {started}", flush=True)
    await asyncio.sleep(sleep_seconds)
    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[depth-limited] step {i} END   at {finished}", flush=True)
    return f"step {i} done"


@env.task
async def submit_one(i: int, sleep_seconds: int) -> str:
    """Try to submit one child task; report whether it was accepted or rejected.
    We catch the submission error here so the parent run survives partial
    rejections — otherwise the first depth-exceeded would fail the whole run."""
    try:
        return await step(i, sleep_seconds)
    except Exception as exc:
        msg = str(exc)
        print(f"[depth-limited] step {i} REJECTED: {msg}", flush=True)
        return f"step {i} rejected: {msg}"


@env.task
async def main(count: int = 8, sleep_seconds: int = 8) -> list[str]:
    """Submit `count` children in a burst against a depth-5 queue. Some are
    admitted; some are rejected with backpressure. Both outcomes are visible
    in the returned list."""
    return list(flyte.map(partial(submit_one, sleep_seconds=sleep_seconds), list(range(count))))


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, count=8, sleep_seconds=8)
    print(run.name)
    print(run.url)
    run.wait()
