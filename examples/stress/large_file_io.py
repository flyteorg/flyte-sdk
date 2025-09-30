import time
from typing import Tuple

import flyte
import flyte.io
import flyte.storage

env = flyte.TaskEnvironment(
    "large_file_io",
    resources=flyte.Resources(
        cpu=4,
        memory="16Gi",
    ),
)


@env.task(cache="auto")
async def create_large_file(size_gigabytes: int = 5) -> flyte.io.File:
    f = flyte.io.File.new_remote()
    chunk_size = 1024 * 1024
    async with f.open("wb", block_size=chunk_size) as fp:
        chunk = b"\0" * chunk_size
        for _ in range(size_gigabytes * 1024):
            await fp.write(chunk)
    return f


@env.task
async def read_large_file(f: flyte.io.File) -> Tuple[int, float]:
    total_bytes = 0
    chunk_size = 1024 * 1024
    start = time.time()
    read = 0
    async with f.open("rb", block_size=chunk_size) as fp:
        while _ := await fp.read():
            read += 1

    end = time.time()
    total = end - start
    print(f"Read {total_bytes} bytes in {total:.2f} seconds ({total_bytes / total / (1024 * 1024):.2f} MiB/s)")
    return total_bytes, total


@env.task
async def main(size_gigabytes: int = 5) -> Tuple[int, float]:
    large_file = await create_large_file(size_gigabytes)
    return await read_large_file(large_file)


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    r = flyte.run(main, 2)
    print(r.url)
