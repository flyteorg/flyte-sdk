"""
Call a deployed reusable task by reference, and give one call a different warm pool.

You only need the task's name, not its source. Overriding the reuse policy at call time does not redeploy
anything; the overridden call lands on its own warm pool. Here that pool runs bigger pods.

    flyte deploy reusable_workers.py env    # once
    python reusable_override.py
"""

import asyncio

import flyte
import flyte.remote

env = flyte.TaskEnvironment(name="reusable_override_driver")

square = flyte.remote.Task.get("reusable_workers.square", auto_version="latest")


@env.task
async def main(n: int) -> list[int]:
    # These calls use the deployed warm pool (1-2 replicas, 500Mi).
    small = await asyncio.gather(*(square(x=x) for x in range(n)))

    # This call gets its own warm pool: 2-4 replicas of 4Gi pods, kept warm for 10 minutes.
    big_square = await square.override.aio(
        short_name="big_square",  # shows up under this name in the UI
        reusable=flyte.ReusePolicy(replicas=(2, 4), idle_ttl=600),
        resources=flyte.Resources(memory="4Gi", cpu=2),
    )
    big = await big_square(x=1_000)

    return [*small, big]


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, n=10)
    print(run.url)
    run.wait()
