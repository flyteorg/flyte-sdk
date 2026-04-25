import asyncio
import logging
from typing import List

import flyte

env = flyte.TaskEnvironment(
    name="hello_world",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    queue="dogfood-1"
)


@env.task(queue="dogfood-2")
async def say_hello(data: str, lt: List[int]) -> str:
    print(f"Hello, world! - {flyte.ctx().action}")
    return f"Hello {data} {lt}"


@env.task(queue="dogfood-2")
async def square(i: int = 10) -> int:
    print(flyte.ctx().action)
    return i * i


@env.task(entrypoint=True)
async def say_hello_nested(data: str = "default string", n: int = 10) -> str:
    print(f"Hello, nested! - {flyte.ctx().action}")
    coros = []
    for i in range(n):
        if i % 2 == 1:
            coros.append(square.override(queue="dogfood-1")(i=i))
        else:
            coros.append(square(i=i))

    vals = await asyncio.gather(*coros)
    return await say_hello(data=data, lt=vals)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(
        log_level=logging.DEBUG,
        env_vars={"KEY": "V"},
        labels={"Label1": "V1"},
        annotations={"Ann": "ann"},
        overwrite_cache=True,
        interruptible=False,
    ).run(say_hello_nested, data="hello world", n=10)
    print(run.name)
    print(run.url)
    run.wait()
    # print(run.outputs())
