import asyncio

import flyte

env = flyte.TaskEnvironment(
    name="blackhole-test",
    env_vars={"LOG_LEVEL": "info"},  # surfaces "Successfully launched action" lines
)


@env.task
async def child(i: int) -> int:
    return i


@env.task
async def parent(n: int = 150, pause_s: float = 5.0) -> int:
    total = 0
    for i in range(n):
        total += await child(i)
        await asyncio.sleep(pause_s)
    return total


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(parent)
    print(run.name, run.url)
