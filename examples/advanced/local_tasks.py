import asyncio
from typing import AsyncGenerator, AsyncIterator, Tuple

import flyte

env = flyte.TaskEnvironment(name="traces", image=flyte.Image.from_debian_base())


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

    print(vals, flush=True)
    print(vals2, flush=True)
    return e1, await do_echo(" ----- ".join(vals)), await do_echo(" ----- ".join(vals2))


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    a = flyte.run(main, "hello world")
    print(a.url)
