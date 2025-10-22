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

from flyte.extras import ContainerTask

env = flyte.TaskEnvironment(
    "file_io_benchmark",
    resources=flyte.Resources(
        cpu=4,
        memory="16Gi",
    ),
    image=flyte.Image.from_debian_base(name="io-benchmarker")
)

old_env = flyte.TaskEnvironment(
    "file_io_benchmark_old",
    resources=flyte.Resources(
        cpu=4,
        memory="16Gi",
    ),
    image=flyte.Image.from_base("ghcr.io/flyteorg/flyte:py3.13-v2.0.0b24"),
)

s5cmd_image = (
    flyte.Image.from_base("public.ecr.aws/aws-cli/aws-cli:latest").clone(name="test-s5cmd")
    .with_commands([
        "wget -q https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz",
        "tar -xzf s5cmd_2.2.2_Linux-64bit.tar.gz",
        "mv s5cmd /usr/local/bin/",
        "rm s5cmd_2.2.2_Linux-64bit.tar.gz",
        "chmod +x /usr/local/bin/s5cmd"
    ])
)

s5cmd_env = flyte.TaskEnvironment(
    "s5cmd_benchmark",
    resources=flyte.Resources(
        cpu=4,
        memory="16Gi",
    ),
    image=s5cmd_image
)

@env.task  # (cache=flyte.Cache(behavior="override", version_override="v2"))
async def create_file(size_megabytes: int = 5120) -> flyte.io.File:
    f = flyte.io.File.new_remote()
    chunk_size = 1024 * 1024
    async with f.open("wb", block_size=chunk_size) as fp:
        chunk = b"\0" * chunk_size
        for _ in range(size_megabytes):
            await fp.write(chunk)
    print(f"Streamed file to {f.path}", flush=True)
    fs = flyte.storage.get_underlying_filesystem("s3")
    await asyncio.sleep(2)
    try:
        print(fs.info(f.path), flush=True)
        print("Successfully got file metadata", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("hanging for debug", flush=True)
        await asyncio.sleep(2000)

    return f


@env.task
async def create_dir_with_files(num_files: int = 1000, size_megabytes: int = 5) -> flyte.io.Dir:
    """Create a directory with multiple files of specified size"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        chunk_size = 1024 * 1024
        chunk = b"\0" * chunk_size

        print(f"Creating {num_files} files of {size_megabytes}MB each in {tmpdir}", flush=True)

        for i in range(num_files):
            file_path = tmppath / f"file_{i:04d}.bin"
            with open(file_path, "wb") as fp:
                for _ in range(size_megabytes):
                    fp.write(chunk)

            if (i + 1) % 100 == 0:
                print(f"Created {i + 1}/{num_files} files", flush=True)

        print(f"Uploading directory to remote storage...", flush=True)
        d = await flyte.io.Dir.from_local(tmppath)
        print(f"Uploaded directory to {d.path}", flush=True)
        return d


async def download(f: flyte.io.File, hang: bool = False) -> Tuple[int, float]:
    """
    Shared core download logic used by both new and old SDK task.
    """
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


async def download_dir(d: flyte.io.Dir, hang: bool = False) -> Tuple[int, float]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        print(f"Will download directory from {d.path} to {tmp_path}", flush=True)
        if hang:
            # This is debugging to exec into the container to monitor it.
            loop = asyncio.get_running_loop()
            waiter = asyncio.Event()
            args = ()

            loop.add_signal_handler(signal.SIGUSR2, waiter.set, *args)
            print(f"Hanging until SIGUSR2 is received... Run 'kill -USR2 {os.getpid()}' to continue", flush=True)
            signal.pause()

        start = time.time()

        await d.download(tmp_path)

        end = time.time()
        total = end - start

        # Calculate total bytes
        total_bytes = 0
        for root, dirs, files in os.walk(tmp_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_bytes += os.path.getsize(file_path)

        print(f"Read {total_bytes} bytes in {total:.2f} seconds ({total_bytes / total / (1024 * 1024):.2f} MiB/s)")

        return total_bytes, total


@env.task
async def read_large_file_new(f: flyte.io.File, hang: bool = False) -> Tuple[int, float]:
    return await download(f, hang)


@old_env.task
async def read_large_file_old(f: flyte.io.File, hang: bool = False) -> Tuple[int, float]:
    return await download(f, hang)


@env.task
async def read_dir_new(d: flyte.io.Dir, hang: bool = False) -> Tuple[int, float]:
    return await download_dir(d, hang)


# S5cmd benchmarks using ContainerTask
s5cmd_file_task = ContainerTask(
    name="s5cmd_download_file",
    image=s5cmd_image,
    inputs={"remote_path": str, "file_size_mb": int},
    outputs={"duration": float, "throughput_mbps": float},
    command=[
        "/bin/bash",
        "-c",
        """
        set -e
        echo "Starting s5cmd download benchmark for file: $0"
        START=$(date +%s.%N)
        s5cmd --no-sign-request cat "$0" > /tmp/downloaded_file || s5cmd cat "$0" > /tmp/downloaded_file
        END=$(date +%s.%N)
        DURATION=$(echo "$END - $START" | bc)
        SIZE_BYTES=$(stat -f%z /tmp/downloaded_file 2>/dev/null || stat -c%s /tmp/downloaded_file)
        THROUGHPUT=$(echo "scale=2; $SIZE_BYTES / $DURATION / 1024 / 1024" | bc)
        echo "Downloaded $SIZE_BYTES bytes in $DURATION seconds ($THROUGHPUT MiB/s)"
        echo "$DURATION" > /tmp/outputs/duration
        echo "$THROUGHPUT" > /tmp/outputs/throughput_mbps
        """,
        "{{.inputs.remote_path}}",
    ],
)


s5cmd_dir_task = ContainerTask(
    name="s5cmd_download_dir",
    image=s5cmd_image,
    inputs={"remote_path": str, "expected_files": int},
    outputs={"duration": float, "throughput_mbps": float},
    command=[
        "/bin/bash",
        "-c",
        """
        set -e
        echo "Starting s5cmd download benchmark for directory: $0"
        mkdir -p /tmp/download_dir
        START=$(date +%s.%N)
        s5cmd --no-sign-request cp "$0/*" /tmp/download_dir/ || s5cmd cp "$0/*" /tmp/download_dir/
        END=$(date +%s.%N)
        DURATION=$(echo "$END - $START" | bc)
        TOTAL_SIZE=$(du -sb /tmp/download_dir | cut -f1)
        THROUGHPUT=$(echo "scale=2; $TOTAL_SIZE / $DURATION / 1024 / 1024" | bc)
        FILE_COUNT=$(find /tmp/download_dir -type f | wc -l)
        echo "Downloaded $FILE_COUNT files, $TOTAL_SIZE bytes in $DURATION seconds ($THROUGHPUT MiB/s)"
        echo "$DURATION" > /tmp/outputs/duration
        echo "$THROUGHPUT" > /tmp/outputs/throughput_mbps
        """,
        "{{.inputs.remote_path}}",
    ],
)


@env.task
async def main(size_megabytes: int = 5120) -> Tuple[int, float]:
    large_file = await create_file(size_megabytes)
    t1 = asyncio.create_task(read_large_file_new(large_file))
    t2 = asyncio.create_task(read_large_file_old(large_file))
    r1, r2 = await asyncio.gather(t1, t2)
    # r1, = await asyncio.gather(t1)
    # print(r1, r2)
    return r1


@env.task
async def benchmark_all():
    """Comprehensive benchmark comparing all download methods"""
    print("=" * 80)
    print("Starting comprehensive I/O benchmarks")
    print("=" * 80)

    # Test 1: Large single file (5GB)
    print("\n--- Test 1: Single 5GB file ---")
    large_file = await create_file(5120)

    print("Running native Flyte download (new SDK)...")
    bytes_new, time_new = await read_large_file_new(large_file)

    print("Running native Flyte download (old SDK)...")
    bytes_old, time_old = await read_large_file_old(large_file)

    print("Running s5cmd download...")
    s5cmd_result = await s5cmd_file_task(remote_path=large_file.path, file_size_mb=5120)
    s5cmd_time = s5cmd_result["duration"]
    s5cmd_throughput = s5cmd_result["throughput_mbps"]

    print(f"\nResults for 5GB file:")
    print(f"  New SDK: {bytes_new / (1024**3):.2f} GB in {time_new:.2f}s ({bytes_new / time_new / (1024**2):.2f} MiB/s)")
    print(f"  Old SDK: {bytes_old / (1024**3):.2f} GB in {time_old:.2f}s ({bytes_old / time_old / (1024**2):.2f} MiB/s)")
    print(f"  s5cmd:   {s5cmd_time:.2f}s ({s5cmd_throughput:.2f} MiB/s)")

    # Test 2: Directory with 1000 5MB files
    print("\n--- Test 2: Directory with 1000 x 5MB files ---")
    file_dir = await create_dir_with_files(num_files=1000, size_megabytes=5)

    print("Running native Flyte directory download (new SDK)...")
    dir_bytes_new, dir_time_new = await read_dir_new(file_dir)

    print("Running s5cmd directory download...")
    s5cmd_dir_result = await s5cmd_dir_task(remote_path=file_dir.path, expected_files=1000)
    s5cmd_dir_time = s5cmd_dir_result["duration"]
    s5cmd_dir_throughput = s5cmd_dir_result["throughput_mbps"]

    print(f"\nResults for 1000 x 5MB files:")
    print(f"  New SDK: {dir_bytes_new / (1024**3):.2f} GB in {dir_time_new:.2f}s ({dir_bytes_new / dir_time_new / (1024**2):.2f} MiB/s)")
    print(f"  s5cmd:   {s5cmd_dir_time:.2f}s ({s5cmd_dir_throughput:.2f} MiB/s)")

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)



if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    r = flyte.run(main, 5)
    print(r.url)




