import asyncio
import logging

import flyte

env = flyte.TaskEnvironment(
    name="queue_routing",
    resources=flyte.Resources(cpu="500m", memory="250Mi"),
)


@env.task
async def hello_default() -> str:
    print(f"Running on default cluster (dogfood-1) - {flyte.ctx().action}")
    return "from dogfood-1"


@env.task(queue="dogfood-2")
async def hello_dogfood_2() -> str:
    print(f"Running on dogfood-2 - {flyte.ctx().action}")
    return "from dogfood-2"


@env.task
async def route_both() -> dict:
    a, b = await asyncio.gather(hello_default(), hello_dogfood_2())
    return {"dogfood_1": a, "dogfood_2": b}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(log_level=logging.DEBUG).run(route_both)
    print(run.name)
    print(run.url)
    run.wait()
