import asyncio
import logging
import flyte
from typing import List

from my_task_library import flyte_entities

img = flyte.Image.from_debian_base().with_pip_packages("my-task-library>=0.4.0", pre=True).with_local_v2()
env = flyte.TaskEnvironment(
    name="library-consumer-env",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=img,
    depends_on=[flyte_entities.library_environment],
)


@env.task
async def use_library(data: str = "default string", n: int = 3) -> str:
    result = await flyte_entities.library_parent_task(data, n)
    print(f"Result from library: {result}")
    return result + " from consumer"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(
        mode="local",
        log_level=logging.DEBUG,
    ).run(use_library, data="hello world", n=10)
    print(run.name)
    print(run.url)
    run.wait()
    # print(run.outputs())
