from typing import List

import flyte

env = flyte.TaskEnvironment(name="map")


@env.task
async def my_task(x: int) -> str:
    return f"Task {x}"


@env.task
async def main(n: int) -> List[str]:
    """
    Run my_task in parallel for the range of n.
    """
    collect: List[str] = []
    async for i in flyte.map.aio(my_task, range(n), return_exceptions=True):
        if isinstance(i, Exception):
            raise i
        collect.append(i)
    return collect


@env.task
def sync_my_task(x: int) -> str:
    """
    Synchronous version of my_task.
    """
    print(f"Task {x}")
    return f"Task {x}"


@env.task
def sync_main(n: int) -> List[str]:
    """
    Synchronous entry point for the task.
    """
    collect: List[str] = []
    for x in flyte.map(sync_my_task, range(n), return_exceptions=True):
        if isinstance(x, Exception):
            raise x
        collect.append(x)
    return collect


@env.task
async def async_to_sync_main(n: int) -> List[str]:
    collect: List[str] = []
    async for x in flyte.map.aio(sync_my_task, range(n)):
        if isinstance(x, Exception):
            raise x
        collect.append(x)
    return collect


if __name__ == "__main__":
    flyte.init_from_config()
    # flyte.init()
    run = flyte.run(async_to_sync_main, 10)
    print(run.url)
    run = flyte.run(main, 10)
    print(run.url)
    run = flyte.run(sync_main, 10)
    print(run.url)
