import asyncio

import flyte.errors

env = flyte.TaskEnvironment("cancel")


@env.task
async def sleepers(f: float, n: int):
    await asyncio.sleep(f)


@env.task
async def failing_task(f: float):
    raise ValueError("I will fail!")


@env.task
async def main(n: int, f: float):
    sleeping_tasks = []
    for i in range(n):
        sleeping_tasks.append(asyncio.create_task(sleepers(f, i)))

    await asyncio.sleep(f)
    try:
        await failing_task(f)
        await asyncio.gather(*sleeping_tasks)
    except flyte.errors.RuntimeUserError as e:
        if e.code == "ValueError":
            print(f"Recevived Value error, canceling {len(sleeping_tasks)} sleeping tasks ")
            for t in sleeping_tasks:
                t.cancel()
        return


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(main, 30, 10.0))
