import asyncio
import typing

import flyte

env = flyte.TaskEnvironment("udfs")


@env.task
async def add_one_udf(x: int) -> int:
    return x + 1


async def fn_add_two_udf(x: int) -> int:
    return x + 2


@env.task
async def run_udf(x: int, udf: typing.Callable[[int], typing.Awaitable[int]]) -> int:
    return await udf(x)


@env.task
async def main() -> list[int]:
    results_coro_one = []
    results_coro_two = []
    for i in range(5):
        results_coro_one.append(asyncio.create_task(run_udf.override(short_name="run_udf_parent")(i, add_one_udf)))
        results_coro_two.append(asyncio.create_task(run_udf.override(short_name="run_udf_inline")(i, fn_add_two_udf)))

    results_one = await asyncio.gather(*results_coro_one)
    results_two = await asyncio.gather(*results_coro_two)
    return results_one + results_two


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
