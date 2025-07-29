import asyncio
from typing import AsyncGenerator, AsyncIterator, Tuple

import flyte

env = flyte.TaskEnvironment(name="env_2", image=flyte.Image.from_debian_base())


@env.task
async def finish() -> str:
    print(f"Finished! - {flyte.ctx().action}", flush=True)
    return "Finished!"


@env.task
async def print_one_number(i: int = 3):
    print(i)


@env.task
async def print_every_second_fanout() -> str:
    coros = []
    for i in range(100):
        coros.append(print_one_number(i))
        await asyncio.sleep(1)

    await asyncio.gather(*coros)
    return await finish()


@env.task
async def print_every_second(mult: int = 4) -> str:
    print(f"Hello, nested! - {flyte.ctx().action}", flush=True)
    for i in range(1000):
        print(f"{mult} * {i} = {mult * i}", flush=True)
        await asyncio.sleep(1)

    return await finish()


@flyte.trace
async def call_llm(q: str) -> str:
    import random

    return f"{q} - {random.random()}"


@flyte.trace
async def stream_llm(q: str) -> AsyncGenerator[str, None]:
    import random

    for i in range(5):
        yield f"{q} - {random.random()}"
        await asyncio.sleep(1)
    return


@flyte.trace
async def stream_iterate(q: str) -> AsyncIterator[str]:
    for i in range(5):
        yield f"{q} - {i}"


@env.task
async def do_echo(q: str) -> str:
    print(q)
    return q


@env.task
async def main(q: str) -> Tuple[str, str, str]:
    v = await call_llm(q)
    e1 = await do_echo(v)
    vals = []
    d = stream_llm(q)
    async for v2 in d:
        vals.append(v2)

    vals2 = []
    async for v2 in stream_iterate(q):
        vals2.append(v2)
    return e1, await do_echo(" ----- ".join(vals)), await do_echo(" ----- ".join(vals2))
