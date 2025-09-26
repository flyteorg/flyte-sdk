import asyncio
from typing import List

import flyte
import flyte.errors
import flyte.git

env = flyte.TaskEnvironment(
    "memory_override_for_loops_1",
    resources=flyte.Resources(cpu=1, memory="400Mi"),
    cache="disable",
)

MEM_OVERRIDES = ["200Mi", "400Mi", "600Mi", "1000Mi"]


@env.task
async def memory_hogger(x: int) -> int:
    size_mb = (x + 1) * 200  # 200MB per level
    print(f"Allocating {size_mb} MB of memory")

    # Allocate memory (1MB = 1024 * 1024 bytes)
    mem = bytearray(size_mb * 1024 * 1024)

    # Touch memory to ensure it's actually allocated
    for k in range(0, len(mem), 4096):  # touch every page (4KB)
        mem[k] = 1
    return x


async def retry_with_more_mem(x: int) -> int:
    """
    Retry foo with more memory if it fails.
    """
    with flyte.group(f"retry-group-{x}"):
        i = 0
        while i < len(MEM_OVERRIDES):
            try:
                return await memory_hogger.override(
                    cache="disable", resources=flyte.Resources(cpu=1, memory=MEM_OVERRIDES[i])
                )(x)
            except flyte.errors.OOMError as e:
                print(f"OOMError encountered: {e}, retrying with more memory")
                i += 1
                if i >= len(MEM_OVERRIDES):
                    print("No more memory overrides available, giving up")
                    raise e


@env.task
async def main(n: int) -> List[int]:
    """
    Run foo in a nested loop structure.
    """
    coros = []
    for i in range(n):
        coros.append(retry_with_more_mem(i))
    result = await asyncio.gather(*coros, return_exceptions=True)

    for res in result:
        if isinstance(res, Exception):
            print(f"Error encountered: {res}")
            raise res
    return result


if __name__ == "__main__":
    flyte.init_from_config(flyte.git.config_from_root())
    run = flyte.run(main, n=3)
    print(run.name)
    print(run.url)
    run.wait(run)
