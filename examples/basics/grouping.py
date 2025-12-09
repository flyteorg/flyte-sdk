import asyncio
from typing import List

import flyte

env = flyte.TaskEnvironment(name="hello_world")


@env.task
async def double(x: int) -> int:
    return x * 2


@env.task
async def group_dynamic(x: int) -> List[int]:
    vals = {"even": [], "odd": []}
    for x in range(x):
        if x % 2 == 0:
            group = "even"
        else:
            group = "odd"
        vals[group].append(double(x))

    with flyte.group("even-group"):
        o1 = await asyncio.gather(*vals["even"])
    with flyte.group("odd-group"):
        o2 = await asyncio.gather(*vals["odd"])

    return o1 + o2


if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.run(group_dynamic, x=10)
    print(run.url)
    run.wait()
