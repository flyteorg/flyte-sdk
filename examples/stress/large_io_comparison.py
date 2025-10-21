import asyncio
import os
import signal
import tempfile
import time
from typing import Tuple
from pathlib import Path

import flyte
import flyte.io
import flyte.storage

env = flyte.TaskEnvironment(
    "file_io_benchmark",
    resources=flyte.Resources(
        cpu=4,
        memory="16Gi",
    ),
)

old_env = flyte.TaskEnvironment(
    "file_io_benchmark_old",
    resources=flyte.Resources(
        cpu=4,
        memory="16Gi",
    ),
    image=flyte.Image.from_base("ghcr.io/flyteorg/flyte:py3.13-v2.0.0b24"),
)

@env.task(cache=flyte.Cache(behavior="override", version_override="v2"))
async def create_large_file(size_gigabytes: int = 5) -> flyte.io.File:
    f = flyte.io.File.new_remote()
    chunk_size = 1024 * 1024
    async with f.open("wb", block_size=chunk_size) as fp:
        chunk = b"\0" * chunk_size
        for _ in range(size_gigabytes * 1024):
            await fp.write(chunk)
    print(f"Streamed file to {f.path}", flush=True)
    return f


async def download(f: flyte.io.File, hang: bool = False) -> Tuple[int, float]:
    _, tmp_path = tempfile.mkstemp()
    print(f"Will download file from {f.path} to {tmp_path}", flush=True)
    if hang:
        # This is debugging to exec into the container to monitor it.
        loop = asyncio.get_running_loop()
        waiter = asyncio.Event()
        args = ()

        loop.add_signal_handler(signal.SIGUSR2, waiter.set, *args)
        print(f"Hanging until SIGUSR2 is received... Run 'kill -USR2 {os.getpid()}' to continue", flush=True)
        signal.pause()

    start = time.time()

    await f.download(tmp_path)

    end = time.time()
    total = end - start
    total_bytes = os.path.getsize(tmp_path)
    print(f"Read {total_bytes} bytes in {total:.2f} seconds ({total_bytes / total / (1024 * 1024):.2f} MiB/s)")

    return total_bytes, total


@env.task
async def read_large_file_new(f: flyte.io.File, hang: bool = False) -> Tuple[int, float]:
    return await download(f, hang)


@old_env.task
async def read_large_file_old(f: flyte.io.File, hang: bool = False) -> Tuple[int, float]:
    return await download(f, hang)


@env.task
async def main(size_gigabytes: int = 5) -> Tuple[int, float]:
    large_file = await create_large_file(size_gigabytes)
    t1 = asyncio.create_task(read_large_file_new(large_file))
    t2 = asyncio.create_task(read_large_file_old(large_file))
    r1, r2 = await asyncio.gather(t1, t2)
    # r1, = await asyncio.gather(t1)
    # print(r1, r2)
    return r1



if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    r = flyte.run(main, 5)
    print(r.url)




