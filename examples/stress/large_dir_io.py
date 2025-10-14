import asyncio
import os
import pathlib
import signal
import tempfile
import time
from typing import Tuple

import flyte
import flyte.io
import flyte.storage

env = flyte.TaskEnvironment(
    "large_dir_io",
    resources=flyte.Resources(
        cpu=4,
        memory="16Gi",
    ),
)


@env.task(cache="auto")
async def create_large_dir(size_gigabytes: int = 5) -> flyte.io.Dir:
    """
    Create a nested directory structure with multiple files totaling the specified size.
    Creates a structure like:
    - root/
      - file_0.bin (500 MB)
      - file_1.bin (500 MB)
      - nested1/
        - file_0.bin (500 MB)
        - file_1.bin (500 MB)
        - nested2/
          - file_0.bin (500 MB)
          ...
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Calculate how many 500MB files we need
        file_size_mb = 500
        total_files = (size_gigabytes * 1024) // file_size_mb

        # Create nested structure: roughly 1/3 at root, 1/3 in nested1, 1/3 in nested2
        files_per_level = total_files // 3
        remainder = total_files % 3

        chunk_size = 1024 * 1024  # 1 MB
        chunk = b"\0" * chunk_size
        chunks_per_file = file_size_mb

        print(f"Creating {total_files} files of {file_size_mb}MB each ({size_gigabytes}GB total)", flush=True)

        # Root level files
        print(f"Creating {files_per_level + remainder} files at root level...", flush=True)
        for i in range(files_per_level + remainder):
            file_path = os.path.join(tmpdir, f"file_{i}.bin")
            with open(file_path, "wb") as f:
                for _ in range(chunks_per_file):
                    f.write(chunk)

        # Nested level 1
        nested1_dir = os.path.join(tmpdir, "nested1")
        os.makedirs(nested1_dir)
        print(f"Creating {files_per_level} files in nested1/...", flush=True)
        for i in range(files_per_level):
            file_path = os.path.join(nested1_dir, f"file_{i}.bin")
            with open(file_path, "wb") as f:
                for _ in range(chunks_per_file):
                    f.write(chunk)

        # Nested level 2
        nested2_dir = os.path.join(nested1_dir, "nested2")
        os.makedirs(nested2_dir)
        print(f"Creating {files_per_level} files in nested1/nested2/...", flush=True)
        for i in range(files_per_level):
            file_path = os.path.join(nested2_dir, f"file_{i}.bin")
            with open(file_path, "wb") as f:
                for _ in range(chunks_per_file):
                    f.write(chunk)

        # Add a sibling folder to nested1 with a small file
        sibling_dir = os.path.join(tmpdir, "sibling")
        os.makedirs(sibling_dir)
        with open(os.path.join(sibling_dir, "metadata.txt"), "w") as f:
            f.write(f"Total files: {total_files}\nTotal size: {size_gigabytes}GB\n")

        print(f"Uploading directory structure from {tmpdir}...", flush=True)
        d = await flyte.io.Dir.from_local(tmpdir)
        print(f"Uploaded to {d.path}", flush=True)
        return d


@env.task
async def read_large_dir(d: flyte.io.Dir, hang: bool = False) -> Tuple[int, float, int]:
    """
    Download a large directory and verify all files are present.
    Returns: (total_bytes, download_time_seconds, file_count)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        download_path = os.path.join(tmpdir, "downloaded")
        print(f"Will download directory from {d.path} to {download_path}", flush=True)

        if hang:
            # This is debugging to exec into the container to monitor it.
            loop = asyncio.get_running_loop()
            waiter = asyncio.Event()
            args = ()

            loop.add_signal_handler(signal.SIGUSR2, waiter.set, *args)
            print(f"Hanging until SIGUSR2 is received... Run 'kill -USR2 {os.getpid()}' to continue", flush=True)
            signal.pause()

        start = time.time()

        downloaded_path = await d.download(download_path)

        end = time.time()
        total = end - start

        # Calculate total size and count files
        total_bytes = 0
        file_count = 0
        for root, dirs, files in os.walk(downloaded_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_bytes += os.path.getsize(file_path)
                file_count += 1

        await asyncio.sleep(100)
        print(f"Downloaded {file_count} files ({total_bytes / (1024**3):.2f} GB) in {total:.2f} seconds ({total_bytes / total / (1024 * 1024):.2f} MiB/s)")

        # Verify nested structure exists
        assert os.path.exists(os.path.join(downloaded_path, "nested1")), "nested1 directory missing"
        assert os.path.exists(os.path.join(downloaded_path, "nested1", "nested2")), "nested2 directory missing"
        assert os.path.exists(os.path.join(downloaded_path, "sibling")), "sibling directory missing"
        print("✓ Directory structure verified", flush=True)

        return total_bytes, total, file_count


@env.task
async def main(size_gigabytes: int = 5) -> Tuple[int, float, int]:
    large_dir = await create_large_dir(size_gigabytes)
    return await read_large_dir(large_dir, hang=False)


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    r = flyte.run(main, 5)
    print(r.url)
