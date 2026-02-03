import asyncio

import flyte

env = flyte.TaskEnvironment("env")


@env.task
async def maybe_take_a_while(x: int) -> int:
    await asyncio.sleep(x)
    return x * x


@env.task
async def fanout_calculation(ls: list[int]) -> int:
    coros = []
    with flyte.group("fanout-with-long-action"):
        for i in ls:
            coros.append(maybe_take_a_while(i))
        results = await asyncio.gather(*coros, return_exceptions=True)
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                print(f"Task {i} failed with {type(res).__name__}: {res}")
            else:
                print(f"Task {i} succeeded with result: {res}")
    return sum(r for r in results if not isinstance(r, Exception))


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(fanout_calculation, [1, 2, 3, 10, 100])
    print(r.url)
