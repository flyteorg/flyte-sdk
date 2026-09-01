"""Exercise RESOURCE_EXHAUSTED handling against a depth-limited queue.

Run with a queue configured with run_concurrency=1, action_concurrency=2,
and depth=5:

    flyte run --queue <queue> examples/cluster_drain/resource_exhausted.py main
"""

import asyncio
import socket
from datetime import datetime, timezone

import flyte

env = flyte.TaskEnvironment(
    name="resource-exhausted",
    resources=flyte.Resources(cpu=1, memory="256Mi"),
)


@env.task
async def hold(i: int, seconds: int) -> int:
    host = socket.gethostname()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"hold {i}: started on {host} at {started}", flush=True)
    await asyncio.sleep(seconds)
    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"hold {i}: finished on {host} at {finished}", flush=True)
    return i


@env.task
async def main(count: int = 8, seconds: int = 5) -> list[int]:
    tasks: list[asyncio.Task[int]] = []
    for i in range(count):
        tasks.append(asyncio.create_task(hold(i, seconds)))
        await asyncio.sleep(0.5)
    results = await asyncio.gather(*tasks)
    expected = list(range(count))
    if results != expected:
        raise AssertionError(f"expected {expected}, got {results}")
    print(f"all {count} actions completed", flush=True)
    return results


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, count=8, seconds=5)
    print(run.name, run.url)
    run.wait()
