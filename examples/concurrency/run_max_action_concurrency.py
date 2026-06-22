import asyncio

import flyte

env = flyte.TaskEnvironment(name="max_action_concurrency")


@env.task
async def step(i: int) -> int:
    await asyncio.sleep(5)
    return i


@env.task
async def main(n: int = 10) -> list[int]:
    tasks = [step(i) for i in range(n)]
    results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(max_action_concurrency=3).run(main, n=10)
    print(run.url)
    # run.wait()
