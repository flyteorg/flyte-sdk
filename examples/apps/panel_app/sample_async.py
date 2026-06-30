import asyncio

import flyte

env = flyte.TaskEnvironment(name="async-example")


# flyte fully supports async Python
@env.task
async def process_value(i: int) -> int:
    # Simulate async computation (network call, database query, etc.)
    await asyncio.sleep(0.2)
    return i * 2


@env.task
async def aggregate_values(values: list[int]) -> int:
    # reduce the values by summing
    return sum(values)


@env.task
async def main(count: int) -> int:
    with flyte.group("process-values"):
        values = await asyncio.gather(*(process_value(i) for i in range(count)))
    return await aggregate_values(values)
