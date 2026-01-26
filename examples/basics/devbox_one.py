import logging
from typing import List

import flyte

env = flyte.TaskEnvironment(
    name="hello_world",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
)


@env.task
def say_hello(data: str, lt: List[int]) -> str:
    print(f"Hello, world! - {flyte.ctx().action}")
    return f"Hello {data} {lt}"


@env.task
def square(i: int = 3) -> int:
    print(flyte.ctx().action)
    return i * i


@env.task
def say_hello_nested(data: str = "default string", n: int = 3) -> str:
    print(f"Hello, nested! - {flyte.ctx().action}")
    # coros = []
    # for i in range(n):
    #     coros.append(square(i=i))
    #
    # vals = await asyncio.gather(*coros)
    say_hello(data=data, lt=[1])
    say_hello(data=data, lt=[1])
    print("done gathering hellos")
    return "finished run"


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
