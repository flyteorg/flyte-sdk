import threading
from typing import List

import flyte
from flyte import Cache

env = flyte.TaskEnvironment(name="ex-cache")


@env.task(cache=Cache(behavior="auto"))
async def t1_auto(data: str, lt: List[int]) -> str:
    print("In t1_auto: this should run every time this code changes...")
    return f"Data is {data}, {lt=}"


@env.task(cache=Cache(behavior="override", version_override="v2", ignored_inputs="ignore_input"))
async def t2_override(ignore_input: str, i: int = 3) -> int:
    print(f"t2_override only runs when manually set version changes and ignores input {ignore_input}")
    return i * i


@env.task
async def cache_parent(data: str = "default string") -> str:
    print(f"In parent say_hello_nested, {data=} from thread: |{threading.current_thread().name}|")
    squared = []
    squared.append(await t2_override(ignore_input="ignore_1", i=1))
    squared.append(await t2_override(ignore_input="ignore_2", i=2))
    squared.append(await t2_override(ignore_input="ignore_3", i=1))

    t1_result = await t1_auto(data=data, lt=squared)
    await t1_auto(data=data, lt=squared)
    return t1_result


@env.task(cache=Cache(behavior="override", version_override="vparent"))
async def parent_is_cached(data: str = "default string") -> str:
    print(f"In parent say_hello_nested, {data=} from thread: |{threading.current_thread().name}|")
    squared = []
    squared.append(await t2_override(ignore_input="ignore_1", i=1))
    squared.append(await t2_override(ignore_input="ignore_2", i=2))
    squared.append(await t2_override(ignore_input="ignore_3", i=1))

    t1_result = await t1_auto(data=data, lt=squared)
    await t1_auto(data=data, lt=squared)
    return t1_result


if __name__ == "__main__":
    import flyte.storage

    flyte.init(
        endpoint="dns:///localhost:8090",
        insecure=True,
        org="testorg",
        project="testproject",
        domain="development",
        storage=flyte.storage.S3.for_sandbox(),
    )

    result = flyte.with_runcontext(mode="local").run(cache_parent, data="hello world")
    print(result)
