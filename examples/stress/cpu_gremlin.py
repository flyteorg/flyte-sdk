import asyncio
from typing import List, Tuple

import flyte

env = flyte.TaskEnvironment(name="cpu_gremlin")


@env.task
async def sleeper(sleep: float) -> str:
    """
    A task that performs a CPU-intensive operation by calculating the sum of squares.
    """
    print(f"Sleeping for {sleep} seconds...")
    await asyncio.sleep(sleep)
    return f"Slept for {sleep} seconds"


async def cpu_hog() -> int:
    """
    A CPU hog function that performs a large number of calculations,
    returning a 64-bit signed integer result.
    """
    print("Starting CPU hog...")
    total = 0
    max_int64 = 2**63
    for i in range(10**7):
        total = (total + i * i) % max_int64
    print(f"CPU hog completed with total: {total}")
    return total


@env.task
async def main(sleep: float = 1.0, n: int = 10) -> Tuple[List[str], int]:
    """
    A task that fans out to multiple instances of sleeper.
    """
    # Run CPU hog in parallel with sleeper tasks
    cpu_hog_task = asyncio.create_task(cpu_hog())
    print("CPU hog task started...", flush=True)

    results = []
    for i in range(n):
        results.append(asyncio.create_task(sleeper(sleep=sleep)))

    print(f"Launching {n} sleeper tasks with {sleep} seconds each...", flush=True)
    await asyncio.sleep(0.2)  # Allow some time for tasks to start

    v = await asyncio.gather(*results)
    print("All sleeper tasks completed.", flush=True)
    return v, await cpu_hog_task


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, sleep=5.0, n=50)  # Adjust the number of sleeper tasks as needed
    print(r.url)
