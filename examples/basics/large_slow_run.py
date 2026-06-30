import asyncio

import flyte

env = flyte.TaskEnvironment("large-slow-run")


@env.task
async def sleeper():
    print("I am going for my first sleep!!!!!!!", flush=True)
    await asyncio.sleep(1)
    print("I am going for my second sleep!!!!!!!", flush=True)
    await asyncio.sleep(1)
    print("I am done....", flush=True)


@env.task
async def parallel_main(n: int):
    coros = []
    for i in range(n):
        coros.append(sleeper())
    await asyncio.gather(*coros)


@env.task
async def main(n: int):
    for i in range(n):
        await sleeper()
