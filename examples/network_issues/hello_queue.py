import asyncio

import flyte

# Also using this for basic queue drain testing/draining -> drained

env_child = flyte.TaskEnvironment(
    name="queue-drain-child",
    # queue="drain-test-a1",
    queue="drain-test-b4",
)

env = flyte.TaskEnvironment(name="queue-drain", depends_on=[env_child])


@env_child.task
async def child(i: int) -> int:
    return i


@env.task
async def parent(n: int = 150, pause_s: float = 5.0) -> int:
    total = 0
    for i in range(n):
        total += await child(i)
        await asyncio.sleep(pause_s)
    return total


@env_child.task
async def long_running_child(i: int) -> int:
    await asyncio.sleep(300)
    return i


@env.task
async def long_running_parent():
    total = 0
    for i in range(2):
        total += await long_running_child(i)
    return total


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(parent)
    print(run.name, run.url)
