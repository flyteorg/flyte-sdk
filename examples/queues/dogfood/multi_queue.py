"""Two queues, independent caps: dogfood-pool (100) vs dogfood-1-low-pri (10).

Same fan-out into each; the low-pri side finishes later because its cap is lower.
"""

import asyncio
from datetime import datetime

import flyte

env_pool = flyte.TaskEnvironment(name="queues_dogfood_multi_pool", resources=flyte.Resources(memory="200Mi"))
env_low = flyte.TaskEnvironment(
    name="queues_dogfood_multi_low", resources=flyte.Resources(memory="200Mi"), depends_on=[env_pool]
)


# { org: dogfood, name: dogfood-pool,
# priority: 50, fairnessAlgorithm: 1, clusters: [dogfood-1, dogfood-2],
# maxActionConcurrency: 100 }
@env_pool.task(queue="dogfood-pool")
async def step_pool(i: int, sleep_seconds: int) -> str:
    print(f"[pool] step {i} START {datetime.utcnow().isoformat(timespec='seconds')}", flush=True)
    await asyncio.sleep(sleep_seconds)
    print(f"[pool] step {i} END   {datetime.utcnow().isoformat(timespec='seconds')}", flush=True)
    return f"pool:{i}"


# { org: dogfood, name: dogfood-1-low-pri,
# priority: 20, fairnessAlgorithm: 1, clusters: [dogfood-1],
# maxActionConcurrency: 10, maxRunConcurrency: 2 }
@env_low.task(queue="dogfood-1-low-pri")
async def step_low(i: int, sleep_seconds: int) -> str:
    print(f"[low-pri] step {i} START {datetime.utcnow().isoformat(timespec='seconds')}", flush=True)
    await asyncio.sleep(sleep_seconds)
    print(f"[low-pri] step {i} END   {datetime.utcnow().isoformat(timespec='seconds')}", flush=True)
    return f"low:{i}"


# main pinned to dogfood-1.
@env_low.task(queue="dogfood-1")
async def main(count_each: int = 20, sleep_seconds: int = 4) -> dict[str, list[str]]:
    pool_tasks = []
    low_tasks = []
    for i in range(count_each):
        pool_tasks.append(asyncio.create_task(step_pool(i, sleep_seconds)))
        low_tasks.append(asyncio.create_task(step_low(i, sleep_seconds)))
    pool = list(await asyncio.gather(*pool_tasks))
    low = list(await asyncio.gather(*low_tasks))
    return {"pool": pool, "low": low}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, count_each=20, sleep_seconds=4)
    print(run.name)
    print(run.url)
    run.wait()
