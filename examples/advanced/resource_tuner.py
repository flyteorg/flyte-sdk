"""
This example shows how to tune and cache optimial resource configurations for a task.

It builds on the `multi_loops.py` example by adding a tune_memory step that
runs and caches the optimal memory configuration for a task-input combination.
"""

import asyncio

import flyte
import flyte.errors


env = flyte.TaskEnvironment(
    "resource_tuner",
    resources=flyte.Resources(cpu=1, memory="400Mi"),
    cache="disable",
)

MEM_OVERRIDES = ["200Mi", "400Mi", "600Mi", "1000Mi"]


@env.task(cache="disable")
async def memory_hogger(x: int) -> int:
    size_mb = (x + 1) * 200  # 200MB per level
    print(f"Allocating {size_mb} MB of memory")

    # Allocate memory (1MB = 1024 * 1024 bytes)
    mem = bytearray(size_mb * 1024 * 1024)

    # Touch memory to ensure it's actually allocated
    for k in range(0, len(mem), 4096):  # touch every page (4KB)
        mem[k] = 1
    return x


@env.task(cache="auto")
async def tune_memory(x: int) -> str:
    """
    Retry foo with more memory if it fails.
    """
    i = 0
    with flyte.group(f"tune-memory-{x}"):
        while i < len(MEM_OVERRIDES):
            try:
                mem = MEM_OVERRIDES[i]
                await memory_hogger.override(resources=flyte.Resources(cpu=1, memory=mem), cache="disable")(x)
                return mem
            except flyte.errors.OOMError as e:
                print(f"OOMError encountered: {e}, retrying with more memory")
                i += 1
                if i >= len(MEM_OVERRIDES):
                    print("No more memory overrides available, giving up")
                    raise e


@env.task(cache="auto")
async def tuning_step(inputs: list[int]) -> dict[int, str]:
    tuned_memories = await asyncio.gather(*[
        tune_memory.override(short_name=f"tune_memory_{i}")(i) for i in inputs
    ])
    return dict(zip(inputs, tuned_memories))


@env.task
async def main(n: int) -> list[int]:
    """
    Run foo in a nested loop structure.
    """
    inputs = list(range(n))
    tuned_mem = await tuning_step(inputs)

    coros = []
    for i in inputs:
        coros.append(memory_hogger.override(resources=flyte.Resources(cpu=1, memory=tuned_mem[i]), cache="disable")(i))
    result = await asyncio.gather(*coros)
    return result


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    run = flyte.run(main, n=3)
    print(run.name)
    print(run.url)
    run.wait(run)
