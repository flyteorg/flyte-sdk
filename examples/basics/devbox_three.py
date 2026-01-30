import asyncio
import logging
from typing import List

import flyte

env = flyte.TaskEnvironment(
    name="hello_world",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
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
async def square_parent(i: int = 3) -> int:
    print(f"In square_parent {i=}", flush=True)
    return await square(i=1)


@env.task
async def say_hello_parent(data: str = "default string", n: int = 3) -> str:
    print(f"Hello, nested! - {flyte.ctx().action}", flush=True)
    return await say_hello(data=data, lt=[n, n])


@env.task
async def main_task(data: str = "default string", n: int = 3) -> str:
    await say_hello_parent(data=data, n=n)
    await square_parent(n=n)
    return "done"

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(
        mode="local",
        log_level=logging.DEBUG,
        env_vars={"KEY": "V"},
        labels={"Label1": "V1"},
        annotations={"Ann": "ann"},
        overwrite_cache=True,
        interruptible=False,
    ).run(main_task)
    print(run.name)
    print(run.url)
    print(run.outputs())
