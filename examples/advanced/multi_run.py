import asyncio
from typing import List

import flyte
from flyte.remote import Run

env = flyte.TaskEnvironment(name="hello_world")


@env.task
async def double(x: int) -> int:
    tctx = flyte.ctx()
    assert tctx is not None  # always set inside a task
    print(tctx.action)
    print(tctx.raw_data_path)
    print(tctx.data)
    print(tctx.code_bundle)
    print(tctx.checkpoint_paths)
    return x * 2


@env.task
async def double_sync(x: int) -> int:
    return x * 2


async def main() -> List[Run]:
    vals = [
        flyte.run.aio(double, x=1),
        flyte.run.aio(double, x=2),
    ]
    return list(await asyncio.gather(*vals))


if __name__ == "__main__":
    flyte.init_from_config()
    asyncio.run(main())
