import asyncio

import flyte

env = flyte.TaskEnvironment(
    name="traces_fanout",
)


@flyte.trace
async def square(x: int) -> int:
    return x * x


@env.task
async def compute_squares(n: int) -> list[int]:
    with flyte.group("squares"):
        coroutines = []
        for number in range(n):
            coroutines.append(square(number))
        return await asyncio.gather(*coroutines)


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    r = flyte.run(compute_squares, n=1000)
    print(r.url)
