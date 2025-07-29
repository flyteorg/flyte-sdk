import asyncio

import flyte

env = flyte.TaskEnvironment("large_fanout")


@env.task
async def my_task(x: int) -> int:
    """
    A task that simply returns the input integer.
    """
    return x


@env.task
async def main(r: int):
    """
    A task that fans out to multiple instances of my_task.
    """
    results = []
    for i in range(r):
        results.append(my_task(x=i))
    return await asyncio.gather(*results)


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    r = flyte.run(main, r=1000)  # Adjust the number of fanouts as needed
    print(r.url)
