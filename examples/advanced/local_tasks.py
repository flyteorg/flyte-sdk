import asyncio
from typing import AsyncGenerator, AsyncIterator, Tuple

import flyte

idl2 = "git+https://github.com/flyteorg/flyte.git@v2#subdirectory=gen/python"

image = (
    flyte.Image.from_debian_base(install_flyte=False).with_apt_packages("git").with_pip_packages(idl2).with_local_v2()
)

env = flyte.TaskEnvironment(
    name="traces",
    image=image,
    resources=flyte.Resources(cpu=1),
)


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


@env.task
async def parallel_main(q: str) -> list[str]:
    r = []
    for i in range(10):
        r.append(await call_llm(q))
    r.append(await do_echo(" ----- ".join(r)))
    # tasks = [call_llm(q) for _ in range(1000)] + [do_echo(q)]
    # results = await asyncio.gather(*tasks)
    return r


@flyte.trace
async def input_trace(a: str, b: str, c: int):
    await asyncio.sleep(1)
    print("Calling LLM without IO", flush=True)


@flyte.trace
async def output_trace() -> int:
    await asyncio.sleep(1)
    print("Calling LLM without IO", flush=True)
    return 42


@flyte.trace
async def noio_trace():
    await asyncio.sleep(1)
    print("Calling LLM without IO", flush=True)


@env.task
def noio_task():
    print("Running noio_task", flush=True)


@env.task
async def parallel_main_no_io(q: str) -> int:
    print("Starting parallel_main_no_io", flush=True)
    noio_task()
    await input_trace("hello world", "blah", 42)
    a = await output_trace()
    await noio_trace()
    b = await input_output_trace("hello", "world", 42, "blah")
    return a + b


@flyte.trace
async def input_output_trace(a: str, b: str, c: int, d: str) -> int:
    return c


@env.task
async def input_output_task(a: str, b: str, c: int) -> int:
    return await input_output_trace(a, b, c, "hello")


if __name__ == "__main__":
    flyte.init_from_config(log_level="DEBUG")
    a = flyte.run(parallel_main_no_io, "hello world")
    print(a.url)
