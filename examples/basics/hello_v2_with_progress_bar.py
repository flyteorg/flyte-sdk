from tqdm.asyncio import tqdm

import flyte

env = flyte.TaskEnvironment(
    name="hello_v2_with_progress_bar",
)


@env.task()
async def hello_worker(id: int) -> str:
    ctx = flyte.ctx()
    assert ctx is not None
    return f"hello, my id is: {id} and I am being run by Action: {ctx.action}"


@env.task()
async def hello_driver(ids: list[int] = [1, 2, 3]) -> list[str]:
    coros = []
    with flyte.group("fanout-group"):
        for id in ids:
            coros.append(hello_worker(id))

        vals = await tqdm.gather(*coros,desc="Running Tasks")

    return vals


if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.run(hello_driver)
    print(run.name)
    print(run.url)
    run.wait()
