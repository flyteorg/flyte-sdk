import asyncio

import flyte

env = flyte.TaskEnvironment("large-slow-run")


@env.task
async def sleeper():
    await asyncio.sleep(2)


@env.task
async def main(n: int):
    for i in range(n):
        await sleeper()
