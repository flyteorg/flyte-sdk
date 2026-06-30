"""Demonstrate flyte.map with max concurrency of 3.

Only 3 tasks should be in-flight at any time.
Each task sleeps for 2 seconds so you can observe batching in the logs.
"""

import asyncio
from typing import List

import flyte

env = flyte.TaskEnvironment(name="map-concurrency")


@env.task
async def slow_task(x: int) -> str:
    print(f"[start] Task {x}")
    await asyncio.sleep(2)
    print(f"[done]  Task {x}")
    return f"result-{x}"


@env.task
async def main() -> List[str]:
    results: List[str] = []
    async for r in flyte.map.aio(slow_task, range(10), concurrency=3):
        if isinstance(r, Exception):
            raise r
        results.append(r)
    return results


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
