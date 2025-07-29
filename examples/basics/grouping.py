import asyncio
from typing import List

import flyte

env = flyte.TaskEnvironment(name="hello_world")


@env.task
async def double(x: int) -> int:
    return x * 2


@env.task
async def root_wf(x: int) -> List[int]:
    print(x)
    vals = []
    with flyte.group("double-list-1"):
        for x in range(x):
            vals.append(double(x))

        o1 = await asyncio.gather(*vals)

    vals = []
    with flyte.group("double-list-2"):
        for x in range(x):
            vals.append(double(x))

        o2 = await asyncio.gather(*vals)

    return o1 + o2


if __name__ == "__main__":
    import flyte.config

    flyte.init_from_config("../../config.yaml")

    run = flyte.run(root_wf, x=10)
    print(run.url)
    run.wait(run)
