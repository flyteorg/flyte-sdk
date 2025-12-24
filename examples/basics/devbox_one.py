import asyncio
import logging
from typing import List

import flyte

env = flyte.TaskEnvironment(
    name="hello_world",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=flyte.Image.from_base("ghcr.io/jeevb/flyte:py3.12-minimal-08"),
)


@env.task
async def say_hello(data: str, lt: List[int]) -> str:
    print(f"Hello, world! - {flyte.ctx().action}")
    return f"Hello {data} {lt}"


@env.task
async def square(i: int = 3) -> int:
    print(flyte.ctx().action)
    return i * i


@env.task
async def say_hello_nested(data: str = "default string", n: int = 3) -> str:
    print(f"Hello, nested! - {flyte.ctx().action}")
    coros = []
    for i in range(n):
        coros.append(square(i=i))

    vals = await asyncio.gather(*coros)
    return await say_hello(data=data, lt=vals)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(
        log_level=logging.DEBUG,
        env_vars={"KEY": "V", "PYTHONVERBOSE": "1", "PYTHONDEBUG": "1"},
        labels={"Label1": "V1"},
        annotations={"Ann": "ann"},
        overwrite_cache=True,
        interruptible=False,
    ).run(say_hello_nested, data="hello world", n=2)
    print(run.name)
    print(run.url)
    run.wait()
    # print(run.outputs())
