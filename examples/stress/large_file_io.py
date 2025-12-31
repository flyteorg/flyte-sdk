import asyncio
import os
import signal
import tempfile
import time
from pathlib import Path
from typing import Tuple

import flyte
import flyte.io
import flyte.storage
from flyte._image import PythonWheels

controller_dist_folder = Path("/Users/ytong/go/src/github.com/flyteorg/sdk-rust/rs_controller/dist")
wheel_layer = PythonWheels(wheel_dir=controller_dist_folder, package_name="flyte_controller_base")
base = flyte.Image.from_debian_base()
rs_controller_image = base.clone(addl_layer=wheel_layer)


env = flyte.TaskEnvironment(
    "large_file_io",
    resources=flyte.Resources(
        cpu=4,
        memory="16Gi",
    ),
    image=rs_controller_image,
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
async def read_large_file(f: flyte.io.File, hang: bool = False) -> Tuple[int, float]:
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
async def main(size_gigabytes: int = 5) -> Tuple[int, float]:
    large_file = await create_large_file(size_gigabytes)
    return await read_large_file(large_file, hang=False)


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, 5)
    print(r.url)
