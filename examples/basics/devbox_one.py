import asyncio
import logging
from typing import List

import flyte

env = flyte.TaskEnvironment(
    name="hello_world",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=flyte.Image.from_debian_base()
    .with_apt_packages("git")
    .with_pip_packages(
        "flyteidl2 @ git+https://github.com/flyteorg/flyte.git@jeev/connectrpc-integration#subdirectory=gen/python",
    ),
)


@env.task
async def say_hello(data: str, lt: List[int]) -> str:
    print(f"Hello, world! - {flyte.ctx().action}")
    return f"Hello {data} {lt}"


@env.task
async def square(i: int = 3) -> int:
    print(flyte.ctx().action)
    return i * i


def _print_versions():
    from importlib.metadata import version
    for pkg in ("flyte", "connectrpc", "flyteidl2"):
        print(f"{pkg}=={version(pkg)}")


@env.task
async def say_hello_nested(data: str = "default string", n: int = 3) -> str:
    _print_versions()
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
