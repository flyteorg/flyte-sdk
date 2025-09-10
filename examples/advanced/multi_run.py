import asyncio
from typing import List

import flyte

env = flyte.TaskEnvironment(name="hello_world")


@env.task
async def double(x: int) -> int:
    print(flyte.ctx().action)
    print(flyte.ctx().raw_data_path)
    print(flyte.ctx().data)
    print(flyte.ctx().code_bundle)
    print(flyte.ctx().checkpoints)
    return x * 2


@env.task
async def double_sync(x: int) -> int:
    return x * 2


async def main() -> List[int]:
    vals = [
        flyte.run.aio(double, x=1),
        flyte.run.aio(double, x=2),
    ]
    return await asyncio.gather(*vals)


if __name__ == "__main__":
    flyte.init_from_config()
    asyncio.run(main())
