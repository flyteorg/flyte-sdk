import asyncio
from typing import List

import flyte

env = flyte.TaskEnvironment(name="hello_world")


@env.task
async def say_hello(data: str, lt: List[int]) -> str:
    return f"Hello {data} {lt}"


@env.task
async def square(i: int = 3) -> int:
    return i * i


@env.task
async def say_hello_nested(data: str = "default string") -> str:
    coros = []
    for i in range(3):
        coros.append(square(i=i))

    vals = await asyncio.gather(*coros)
    return await say_hello(data=data, lt=vals)


def test_run_local_controller():
    flyte.init_from_config(None)

    result = flyte.with_runcontext(mode="local").run(say_hello_nested, data="hello world")
    assert result.outputs()[0] == "Hello hello world [0, 1, 4]"


def test_run_pure_function():
    result = asyncio.run(say_hello_nested(data="hello world"))
    assert result == "Hello hello world [0, 1, 4]"
