import asyncio

import flyte

env = flyte.TaskEnvironment(name="repeated_tasks")


@env.task
async def repeated_task() -> int:
    return 42


@env.task
async def main_task() -> int:
    repeater = []
    for _ in range(110):
        repeater.append(repeated_task())
    vals = await asyncio.gather(*repeater)
    return sum(vals)


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")

    run = flyte.run(main_task)
    print(run)
