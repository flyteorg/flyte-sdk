import time
from typing import Tuple
import asyncio

import flyte
import flyte.io
import flyte.storage

env = flyte.TaskEnvironment(
    "large_file_io",
    resources=flyte.Resources(
        cpu=4,
        memory="32Gi",
    )
)


@env.task(cache="auto")
async def create_large_file(size_gigabytes: int = 5) -> flyte.io.File:
    f = flyte.io.File.new_remote()

    async with f.open("wb") as fp:
        chunk = b'\0' * (1024 * 1024)  # 1 MiB chunk
        for _ in range(size_gigabytes * 1024):
            fp.write(chunk)
    print(f"Path is {f.path=}", flush=True)
    print("sleeping")
    await asyncio.sleep(40000)
    return f


@env.task
async def read_large_file(f: flyte.io.File) -> Tuple[int, float]:
    total_bytes = 0
    chunk_size = 1024 * 1024
    print(f"Reading {f.name} from {f.name}", flush=True)
    start = time.time()
    read = 0
    async for c in flyte.storage.get_stream(f.path, chunk_size):
       read += 1

    end = time.time()
    total = end - start
    print(f"Read {total_bytes} bytes in {total:.2f} seconds ({total_bytes / total / (1024 * 1024):.2f} MiB/s)")
    return total_bytes, total


@env.task
async def main(size_gigabytes: int = 5) -> Tuple[int, float]:
    large_file = await create_large_file(size_gigabytes)
    # return await read_large_file(large_file)

if __name__ == "__main__":
    import flyte.git
    flyte.init_from_config(flyte.git.config_from_root())
    # r = flyte.run(main, 5)
    r = flyte.run(create_large_file)
    print(r.url)