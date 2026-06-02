"""dogfood-1-low-pri caps runs at 2. Submit 5, at most 2 mains run at once.

Children use default routing (dogfood-1) so they don't count against the run cap.

    python examples/queues/dogfood/serial_runs.py        # 5 runs
    NUM_RUNS=8 python examples/queues/dogfood/serial_runs.py
"""

import asyncio
import os
from datetime import datetime

import flyte

env = flyte.TaskEnvironment(name="queues_dogfood_serial_runs", resources=flyte.Resources(memory="200Mi"))


@env.task
async def child(i: int, sleep_seconds: int) -> str:
    print(f"child {i} START {datetime.utcnow().isoformat(timespec='seconds')}", flush=True)
    await asyncio.sleep(sleep_seconds)
    print(f"child {i} END   {datetime.utcnow().isoformat(timespec='seconds')}", flush=True)
    return f"child{i}"


# { org: dogfood, name: dogfood-1-low-pri,
# priority: 20, fairnessAlgorithm: 1, clusters: [dogfood-1],
# maxActionConcurrency: 10, maxRunConcurrency: 2 }
# Root pinned to the queue (cluster dogfood-1). Run cap gates overlap.
@env.task(queue="dogfood-1")
async def main(fan_out: int = 20, child_sleep_seconds: int = 2) -> list[str]:
    print(f"main START {datetime.utcnow().isoformat(timespec='seconds')} fan_out={fan_out}", flush=True)
    tasks = []
    for i in range(fan_out):
        tasks.append(asyncio.create_task(child(i, child_sleep_seconds)))
    results = list(await asyncio.gather(*tasks))
    print(f"main END   {datetime.utcnow().isoformat(timespec='seconds')}", flush=True)
    return results


async def submit_concurrent_runs(num_runs: int, fan_out: int, sleep_s: int):
    runs = await asyncio.gather(
        *[flyte.run.aio(main, fan_out=fan_out, child_sleep_seconds=sleep_s) for _ in range(num_runs)]
    )
    for i, r in enumerate(runs):
        print(f"  [{i}] {r.name} {r.url}")
    await asyncio.gather(*[r.wait.aio() for r in runs])
    print("all runs done")


if __name__ == "__main__":
    flyte.init_from_config("/Users/ytong/.flyte/dogfood.staging.yaml")
    asyncio.run(
        submit_concurrent_runs(
            int(os.environ.get("NUM_RUNS", "5")),
            int(os.environ.get("FAN_OUT", "20")),
            int(os.environ.get("CHILD_SLEEP", "2")),
        )
    )
