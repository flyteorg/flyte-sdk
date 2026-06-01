import flyte

env = flyte.TaskEnvironment("trace-recur")


@flyte.trace
async def dive(x: int, max_level: int) -> int:
    if max_level == x:
        return x
    return await dive(x + 1, max_level)


@env.task
async def main_outer(max_level: int) -> int:
    return await main(max_level + 1)


@env.task
async def main(max_level: int) -> int:
    return await dive(0, max_level)
