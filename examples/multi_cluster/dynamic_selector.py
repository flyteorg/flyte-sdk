import asyncio
import random

import flyte
import flyte.errors

env = flyte.TaskEnvironment(
    "dynamic-selector",
)


@env.task
async def worker(x: int, cluster: str) -> int:
    return x


@flyte.trace
async def next_cluster() -> str:
    return random.choice(["a", "b", "c"])


async def assign(x: int, max_retries: int = 3) -> int:
    """
    In case of assignment fails because of timeout, we will reassign to a different cluster.
    Args:
        x: int
        max_retries: int
    Returns: result
    """
    retries = 0
    while True:
        cluster = await next_cluster()
        try:
            return await worker.override(queue=cluster)(x, cluster)
        except flyte.errors.TaskTimeoutError    :
            retries += 1
            if retries >= max_retries:
                raise flyte.errors.TaskTimeoutError


@env.task
async def driver(n: int) -> int:
    coros = []
    for i in range(n):
        coros.append(assign(i))
    results = await asyncio.gather(*coros, return_exceptions=True)
    for r in results:
        if isinstance(r, flyte.errors.TaskTimeoutError):
            raise r
    return sum(results)


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(driver, 10)
    print(r.url)
