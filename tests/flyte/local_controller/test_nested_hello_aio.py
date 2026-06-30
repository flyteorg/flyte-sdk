import asyncio
import threading
from typing import List

import flyte

env = flyte.TaskEnvironment(name="hello_world")


@env.task
def say_hello(data: str, lt: List[int]) -> str:
    print(f"In child say_hello, {data=}, {lt=} from thread: |{threading.current_thread().name}|")
    return f"Hello {data} {lt}"


@env.task
def square(i: int = 3) -> int:
    print(f"In square, {i=} from thread: |{threading.current_thread().name}|")
    return i * i


@env.task
async def say_hello_nested(data: str = "default string") -> str:
    print(f"In parent say_hello_nested, {data=} from thread: |{threading.current_thread().name}|")
    coros = []
    for i in range(3):
        coros.append(square.aio(i=i))

    squared = await asyncio.gather(*coros)

    return say_hello(data=data, lt=squared)


def test_run_local_controller():
    flyte.init_from_config(None)

    result = flyte.with_runcontext(mode="local").run(say_hello_nested, data="hello world")
    assert result.outputs()[0] == "Hello hello world [0, 1, 4]"


def test_run_pure_function():
    result = asyncio.run(say_hello_nested(data="hello world"))
    assert result == "Hello hello world [0, 1, 4]"
