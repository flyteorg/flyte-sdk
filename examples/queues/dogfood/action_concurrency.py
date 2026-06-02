"""dogfood-1-low-pri caps actions at 10. Submit 30, at most 10 run at once."""

import asyncio
from datetime import datetime

import flyte

env = flyte.TaskEnvironment(name="queues_dogfood_action_concurrency", resources=flyte.Resources(memory="200Mi"))


# { org: dogfood, name: dogfood-1-low-pri,
# priority: 20, fairnessAlgorithm: 1, clusters: [dogfood-1],
# maxActionConcurrency: 10, maxRunConcurrency: 2 }
@env.task(queue="dogfood-1-low-pri")
async def step(i: int, sleep_seconds: int) -> str:
    print(f"[low-pri] step {i} START {datetime.utcnow().isoformat(timespec='seconds')}", flush=True)
    await asyncio.sleep(sleep_seconds)
    print(f"[low-pri] step {i} END   {datetime.utcnow().isoformat(timespec='seconds')}", flush=True)
    return f"step {i} done"


@env.task(queue="dogfood-1")
async def main(count: int = 30, sleep_seconds: int = 2) -> list[str]:
    tasks = []
    for i in range(count):
        tasks.append(asyncio.create_task(step(i, sleep_seconds)))
    return list(await asyncio.gather(*tasks))


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, count=30, sleep_seconds=4)
    print(run.name)
    print(run.url)
    run.wait()
