import asyncio

import flyte

env = flyte.TaskEnvironment(name="leaky_coroutines")


@env.task
async def do_something():
    print("Doing something")
    await asyncio.sleep(5)
    print("Finished doing something")


@env.task
async def sleep_for(seconds: int):
    print(f"Sleeping for {seconds} seconds")
    try:
        await asyncio.sleep(seconds)
        await do_something()
    except asyncio.CancelledError:
        print(f"Slept for {seconds} seconds was cancelled")
        return
    print(f"Finished sleeping for {seconds} seconds")


@env.task
async def main(seconds: int):
    print("Starting main coroutine")
    t1 = asyncio.create_task(sleep_for(seconds))
    await asyncio.sleep(10)
    print(f"Finished main coroutine, leaked {t1}")


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, seconds=30)
    print(run.url)
    run.wait()
    print("Run completed")
