import asyncio

import nest_asyncio

import flyte

env = flyte.TaskEnvironment(
    "asyncclient_sync",
    image=flyte.Image.from_debian_base().with_pip_packages("aiohttp", "nest_asyncio"),
)

nest_asyncio.apply()


async def async_in_sync() -> str:
    await asyncio.sleep(1)
    return "done"


@env.task
def call_async() -> str:
    return asyncio.run(async_in_sync())


if __name__ == "__main__":

    flyte.init_from_config()
    run = flyte.run(call_async)
    print(run.url)
    run.wait()
