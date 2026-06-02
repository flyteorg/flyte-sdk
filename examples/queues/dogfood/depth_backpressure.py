"""dogfood-pool-depth-lmt admits 200 tasks. Submit 300, ~100 get RESOURCE_EXHAUSTED.

main spawns all 300 step tasks directly and catches the rejections.
"""

import asyncio
from datetime import datetime

import flyte

env = flyte.TaskEnvironment(name="queues_dogfood_depth_backpressure", resources=flyte.Resources(memory="200Mi"))


# { org: dogfood, name: dogfood-pool-depth-lmt,
# priority: 50, fairnessAlgorithm: 1, clusters: [dogfood-1, dogfood-2],
# maxActionConcurrency: 100, maxDepth: 200 }
@env.task(queue="dogfood-pool-depth-lmt")
async def step(i: int, sleep_seconds: int) -> str:
    print(f"[depth] step {i} START {datetime.utcnow().isoformat(timespec='seconds')}", flush=True)
    await asyncio.sleep(sleep_seconds)
    print(f"[depth] step {i} END   {datetime.utcnow().isoformat(timespec='seconds')}", flush=True)
    return f"step {i} done"


@env.task(queue="dogfood-1")
async def main(count: int = 300, sleep_seconds: int = 10) -> list[str]:
    tasks = []
    for i in range(count):
        tasks.append(asyncio.create_task(step(i, sleep_seconds)))
    # return_exceptions=True so rejected submissions don't fail the whole run.
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"[depth] step {i} REJECTED: {r}", flush=True)
            out.append(f"step {i} rejected: {r}")
        else:
            out.append(r)
    return out


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, count=300, sleep_seconds=10)
    print(run.name)
    print(run.url)
    run.wait()
