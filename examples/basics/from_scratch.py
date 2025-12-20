import asyncio

import flyte

env = flyte.TaskEnvironment(
    "from_scratch",
)


@env.task
async def square(x: int) -> int:
    return x * x


@env.task
async def main(n: int) -> int:
    coros = []
    for i in range(n):
        coros.append(square(i))
    results = await asyncio.gather(*coros)
    return sum(results)


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, 10)
    print(r.url)
