import asyncio
import logging

import flyte

env = flyte.TaskEnvironment(
    name="hello_timing",
    resources=flyte.Resources(cpu="500m", memory="300Mi"),
)


@env.task
async def child(i: int) -> int:
    print(f"Child {i} - {flyte.ctx().action}")
    return i


@env.task(entrypoint=True)
async def time_children(n: int = 10) -> None:
    await child(i=0)
    for i in range(1, n + 1):
        await asyncio.sleep(0.25)
        await child(i=i)


@env.task(entrypoint=True)
async def time_children_parallel(n: int = 20) -> None:
    await asyncio.gather(*(child(i=i) for i in range(n)))


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(
        log_level=logging.DEBUG,
        overwrite_cache=True,
        interruptible=False,
    ).run(time_children_parallel, n=20)
    print(run.name)
    print(run.url)
    run.wait()
