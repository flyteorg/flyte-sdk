import asyncio

import flyte

env = flyte.TaskEnvironment(name="hello_world")


@env.task
async def say_hello(name: str):
    print(f"Hello World, {name}")


if __name__ == "__main__":
    asyncio.run(say_hello("human"))
    flyte.init()
    run = flyte.run(say_hello, name="human")
