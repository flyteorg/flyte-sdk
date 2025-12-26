import asyncio
import os
import signal
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import flyte
import flyte.io
import flyte.report
import flyte.storage
from flyte.extras import ContainerTask


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run"""

    bytes: int
    duration: float


# S3fs environment for comparing with v1 flytekit-style downloads
s3fs_env = flyte.TaskEnvironment(
    "s3fs_benchmark",
    resources=flyte.Resources(
        cpu=8,
        memory="23Gi",
    ),
    image=flyte.Image.from_debian_base(name="s3fs-benchmarker").with_pip_packages("s3fs", "fsspec"),
)

s5cmd_image = (
    flyte.Image.from_debian_base(name="s5cmd-benchmark")
    .with_apt_packages("wget", "ca-certificates", "bc")
    .with_commands(
        [
            "wget -q https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz",
            "tar -xzf s5cmd_2.2.2_Linux-64bit.tar.gz",
            "mv s5cmd /usr/local/bin/",
            "rm s5cmd_2.2.2_Linux-64bit.tar.gz",
            "chmod +x /usr/local/bin/s5cmd",
        ]
    )
)

# S5cmd benchmarks using ContainerTask
s5cmd_file_task = ContainerTask(
    name="s5cmd_download_file",
    image=s5cmd_image,
    resources=flyte.Resources(
        cpu=8,
        memory="23Gi",
    ),
    inputs={"remote_path": str, "file_size_mb": int},
    outputs={"duration": float, "throughput_mbps": float},
    command=[
        "/bin/bash",
        "-c",
        """
        set -e
        echo "Starting s5cmd download benchmark for file: $0"
        START=$(date +%s%N)
        s5cmd cp -c 32 "$0" /tmp/downloaded_file
        END=$(date +%s%N)
        DURATION_NS=$((END - START))
        DURATION=$(echo "scale=2; $DURATION_NS / 1000000000" | bc)
        SIZE_BYTES=$(stat -c%s /tmp/downloaded_file)
        THROUGHPUT=$(echo "scale=2; $SIZE_BYTES / $DURATION / 1024 / 1024" | bc)
        echo "Downloaded $SIZE_BYTES bytes in $DURATION seconds ($THROUGHPUT MiB/s)"
        echo "$DURATION" > /var/outputs/duration
        echo "$THROUGHPUT" > /var/outputs/throughput_mbps
        """,
        "{{.inputs.remote_path}}",
    ],
)


s5cmd_dir_task = ContainerTask(
    name="s5cmd_download_dir",
    image=s5cmd_image,
    resources=flyte.Resources(
        cpu=8,
        memory="23Gi",
    ),
    inputs={"remote_path": str, "expected_files": int},
    outputs={"duration": float, "throughput_mbps": float},
    command=[
        "/bin/bash",
        "-c",
        """
        set -e
        echo "Starting s5cmd download benchmark for directory: $0"
        mkdir -p /tmp/download_dir
        START=$(date +%s%N)
        s5cmd cp -c 32 "$0/*" /tmp/download_dir/
        END=$(date +%s%N)
        DURATION_NS=$((END - START))
        DURATION=$(echo "scale=2; $DURATION_NS / 1000000000" | bc)
        TOTAL_SIZE=$(du -sb /tmp/download_dir | cut -f1)
        THROUGHPUT=$(echo "scale=2; $TOTAL_SIZE / $DURATION / 1024 / 1024" | bc)
        FILE_COUNT=$(find /tmp/download_dir -type f | wc -l)
        echo "Downloaded $FILE_COUNT files, $TOTAL_SIZE bytes in $DURATION seconds ($THROUGHPUT MiB/s)"
        echo "$DURATION" > /var/outputs/duration
        echo "$THROUGHPUT" > /var/outputs/throughput_mbps
        """,
        "{{.inputs.remote_path}}",
    ],
)


s5cmd_env = flyte.TaskEnvironment.from_task("s5cmd_env", s5cmd_file_task, s5cmd_dir_task)


env = flyte.TaskEnvironment(
    "file_io_benchmark",
    resources=flyte.Resources(
        cpu=8,
        memory="23Gi",
    ),
    depends_on=[s5cmd_env, s3fs_env],
    image=flyte.Image.from_debian_base(name="io-benchmarker"),
)


@env.task(cache=flyte.Cache(behavior="override", version_override="v3"))
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
    except Exception:
        import traceback

        traceback.print_exc()
        print("hanging for debug", flush=True)
        await asyncio.sleep(2000)

    return f


@env.task(cache=flyte.Cache(behavior="override", version_override="v1"))
async def create_dir_with_files(num_files: int = 1000, size_megabytes: int = 5) -> flyte.io.Dir:
    """Create a directory with multiple files of specified size"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        chunk_size = 1024 * 1024
        chunk = b"\0" * chunk_size

        print(f"Creating {num_files} files of {size_megabytes}MB each in {tmpdir}", flush=True)

        for i in range(num_files):
            file_path = tmppath / f"file_{i:04d}.bin"
            with open(file_path, "wb") as fp:  # noqa: ASYNC230
                for _ in range(size_megabytes):
                    fp.write(chunk)

            if (i + 1) % 100 == 0:
                print(f"Created {i + 1}/{num_files} files", flush=True)

        print("Uploading directory to remote storage...", flush=True)
        d = await flyte.io.Dir.from_local(tmppath)
        print(f"Uploaded directory to {d.path}", flush=True)
        return d


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
async def read_large_file_new(f: flyte.io.File, hang: bool = False, iterations: int = 5) -> List[BenchmarkResult]:
    results = []
    for i in range(iterations):
        print(f"[New SDK File] Run {i + 1}/{iterations}", flush=True)
        bytes_val, duration = await download(f, hang)
        results.append(BenchmarkResult(bytes=bytes_val, duration=duration))
    return results


@env.task
async def read_dir_new(d: flyte.io.Dir, hang: bool = False, iterations: int = 10) -> List[BenchmarkResult]:
    results = []
    for i in range(iterations):
        print(f"[New SDK Dir] Run {i + 1}/{iterations}", flush=True)
        bytes_val, duration = await download_dir(d, hang)
        results.append(BenchmarkResult(bytes=bytes_val, duration=duration))
    return results


# S3fs + fsspec benchmark tasks (similar to v1 flytekit)
async def download_s3fs(remote_path: str, is_directory: bool = False) -> Tuple[int, float]:
    """Download using s3fs directly (v1 flytekit-style)"""
    import s3fs

    _, tmp_path = tempfile.mkstemp() if not is_directory else (None, tempfile.mkdtemp())

    # Parse S3 path
    protocol = remote_path.split("://")[0]
    if protocol != "s3":
        raise ValueError(f"s3fs benchmark only supports s3:// paths, got {protocol}")

    print(f"Will download from {remote_path} to {tmp_path} using s3fs", flush=True)

    # Create s3fs filesystem
    fs = s3fs.S3FileSystem(anon=False)

    start = time.time()

    if is_directory:
        # Download directory
        fs.get(remote_path, tmp_path, recursive=True)
        end = time.time()
        total = end - start

        # Calculate total bytes
        total_bytes = 0
        for root, dirs, files in os.walk(tmp_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_bytes += os.path.getsize(file_path)
    else:
        # Download single file
        fs.get(remote_path, tmp_path)
        end = time.time()
        total = end - start
        total_bytes = os.path.getsize(tmp_path)

    print(f"s3fs: Read {total_bytes} bytes in {total:.2f} seconds ({total_bytes / total / (1024 * 1024):.2f} MiB/s)")

    return total_bytes, total


@s3fs_env.task
async def read_large_file_s3fs(f: flyte.io.File, iterations: int = 10) -> List[BenchmarkResult]:
    results = []
    for i in range(iterations):
        print(f"[s3fs File] Run {i + 1}/{iterations}", flush=True)
        bytes_val, duration = await download_s3fs(f.path, is_directory=False)
        results.append(BenchmarkResult(bytes=bytes_val, duration=duration))
    return results


@s3fs_env.task
async def read_dir_s3fs(d: flyte.io.Dir, iterations: int = 10) -> List[BenchmarkResult]:
    results = []
    for i in range(iterations):
        print(f"[s3fs Dir] Run {i + 1}/{iterations}", flush=True)
        bytes_val, duration = await download_s3fs(d.path, is_directory=True)
        results.append(BenchmarkResult(bytes=bytes_val, duration=duration))
    return results


def generate_benchmark_report(
    file_results_new: List[BenchmarkResult],
    file_results_s3fs: List[BenchmarkResult],
    file_results_s5cmd: List[BenchmarkResult],
    dir_results_new: List[BenchmarkResult],
    dir_results_s3fs: List[BenchmarkResult],
    dir_results_s5cmd: List[BenchmarkResult],
) -> str:
    """Generate HTML report with benchmark results"""

    def calculate_stats(results: List[BenchmarkResult]):
        """Calculate average, min, max from list of BenchmarkResult"""
        times = [r.duration for r in results]
        bytes_vals = [r.bytes for r in results]
        avg_time = sum(times) / len(times)
        avg_bytes = sum(bytes_vals) / len(bytes_vals)
        min_time = min(times)
        max_time = max(times)
        avg_throughput = avg_bytes / avg_time / (1024**2)
        return avg_bytes, avg_time, min_time, max_time, avg_throughput, times

    # Calculate stats for file benchmarks
    file_new_bytes, file_new_time, file_new_min, file_new_max, file_new_tput, file_new_times = calculate_stats(
        file_results_new
    )
    file_s3fs_bytes, file_s3fs_time, file_s3fs_min, file_s3fs_max, file_s3fs_tput, file_s3fs_times = calculate_stats(
        file_results_s3fs
    )
    _, file_s5cmd_time, file_s5cmd_min, file_s5cmd_max, file_s5cmd_tput, file_s5cmd_times = calculate_stats(
        file_results_s5cmd
    )

    # Calculate stats for directory benchmarks
    dir_new_bytes, dir_new_time, dir_new_min, dir_new_max, dir_new_tput, dir_new_times = calculate_stats(
        dir_results_new
    )
    dir_s3fs_bytes, dir_s3fs_time, dir_s3fs_min, dir_s3fs_max, dir_s3fs_tput, dir_s3fs_times = calculate_stats(
        dir_results_s3fs
    )
    _, dir_s5cmd_time, dir_s5cmd_min, dir_s5cmd_max, dir_s5cmd_tput, dir_s5cmd_times = calculate_stats(
        dir_results_s5cmd
    )

    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #555; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .stats {{ font-size: 0.9em; color: #666; }}
            .runs {{ font-size: 0.85em; color: #888; margin-top: 5px; }}
        </style>
    </head>
    <body>
        <h1>I/O Benchmark Results (10 Runs Each)</h1>

        <h2>Test 1: Single 5GB File Download</h2>
        <table>
            <tr>
                <th>Method</th>
                <th>Avg Size (GB)</th>
                <th>Avg Duration (s)</th>
                <th>Min/Max (s)</th>
                <th>Avg Throughput (MiB/s)</th>
                <th>All Runs (s)</th>
            </tr>
            <tr>
                <td>New SDK (Flyte v2)</td>
                <td>{file_new_bytes / (1024**3):.2f}</td>
                <td>{file_new_time:.2f}</td>
                <td>{file_new_min:.2f} / {file_new_max:.2f}</td>
                <td>{file_new_tput:.2f}</td>
                <td class="runs">{", ".join([f"{t:.2f}" for t in file_new_times])}</td>
            </tr>
            <tr>
                <td>s3fs + fsspec (v1 style)</td>
                <td>{file_s3fs_bytes / (1024**3):.2f}</td>
                <td>{file_s3fs_time:.2f}</td>
                <td>{file_s3fs_min:.2f} / {file_s3fs_max:.2f}</td>
                <td>{file_s3fs_tput:.2f}</td>
                <td class="runs">{", ".join([f"{t:.2f}" for t in file_s3fs_times])}</td>
            </tr>
            <tr>
                <td>s5cmd (ContainerTask)</td>
                <td>~5.00</td>
                <td>{file_s5cmd_time:.2f}</td>
                <td>{file_s5cmd_min:.2f} / {file_s5cmd_max:.2f}</td>
                <td>{file_s5cmd_tput:.2f}</td>
                <td class="runs">{", ".join([f"{t:.2f}" for t in file_s5cmd_times])}</td>
            </tr>
        </table>

        <h2>Test 2: Directory with 1000 x 5MB Files</h2>
        <table>
            <tr>
                <th>Method</th>
                <th>Avg Size (GB)</th>
                <th>Avg Duration (s)</th>
                <th>Min/Max (s)</th>
                <th>Avg Throughput (MiB/s)</th>
                <th>All Runs (s)</th>
            </tr>
            <tr>
                <td>New SDK (Flyte v2)</td>
                <td>{dir_new_bytes / (1024**3):.2f}</td>
                <td>{dir_new_time:.2f}</td>
                <td>{dir_new_min:.2f} / {dir_new_max:.2f}</td>
                <td>{dir_new_tput:.2f}</td>
                <td class="runs">{", ".join([f"{t:.2f}" for t in dir_new_times])}</td>
            </tr>
            <tr>
                <td>s3fs + fsspec (v1 style)</td>
                <td>{dir_s3fs_bytes / (1024**3):.2f}</td>
                <td>{dir_s3fs_time:.2f}</td>
                <td>{dir_s3fs_min:.2f} / {dir_s3fs_max:.2f}</td>
                <td>{dir_s3fs_tput:.2f}</td>
                <td class="runs">{", ".join([f"{t:.2f}" for t in dir_s3fs_times])}</td>
            </tr>
            <tr>
                <td>s5cmd (ContainerTask)</td>
                <td>~4.88</td>
                <td>{dir_s5cmd_time:.2f}</td>
                <td>{dir_s5cmd_min:.2f} / {dir_s5cmd_max:.2f}</td>
                <td>{dir_s5cmd_tput:.2f}</td>
                <td class="runs">{", ".join([f"{t:.2f}" for t in dir_s5cmd_times])}</td>
            </tr>
        </table>

        <p><em>All tests run 10 times each in parallel on nodes with 8 CPU, 23Gi memory</em></p>
    </body>
    </html>
    """
    return html


@env.task(report=True)
async def benchmark_all():
    """Comprehensive benchmark comparing all download methods"""
    iterations = 2
    print("=" * 80)
    print(f"Starting comprehensive I/O benchmarks ({iterations} runs each)")
    print("=" * 80)

    # Test 1: Large single file (5GB)
    print("\n--- Test 1: Single 5GB file (10 runs) ---")
    large_file = await create_file(5120)

    print("Running all downloads in parallel (each method runs 10 iterations)...")
    # Run the native tasks (they handle 10 iterations internally)
    t1 = asyncio.create_task(read_large_file_new(large_file, iterations=iterations))
    t2 = asyncio.create_task(read_large_file_s3fs(large_file, iterations=iterations))

    # Run s5cmd 10 times in parallel
    async def run_s5cmd_file_multiple():
        print("[s5cmd File] Starting 10 parallel runs...", flush=True)
        tasks = [s5cmd_file_task(remote_path=large_file.path, file_size_mb=5120) for _ in range(iterations)]
        results = await asyncio.gather(*tasks)
        # Convert s5cmd results (duration, throughput) to BenchmarkResult
        file_size_bytes = 5120 * 1024 * 1024
        return [BenchmarkResult(bytes=file_size_bytes, duration=r[0]) for r in results]

    t3 = asyncio.create_task(run_s5cmd_file_multiple())

    file_results_new, file_results_s3fs, file_results_s5cmd = await asyncio.gather(t1, t2, t3)

    # Print summary
    avg_time_new = sum(r.duration for r in file_results_new) / len(file_results_new)
    avg_time_s3fs = sum(r.duration for r in file_results_s3fs) / len(file_results_s3fs)
    avg_time_s5cmd = sum(r.duration for r in file_results_s5cmd) / len(file_results_s5cmd)

    print("\n5GB file average results:")
    print(f"  New SDK:  {avg_time_new:.2f}s avg")
    print(f"  s3fs:     {avg_time_s3fs:.2f}s avg")
    print(f"  s5cmd:    {avg_time_s5cmd:.2f}s avg")

    # Test 2: Directory with 1000 5MB files
    print("\n--- Test 2: Directory with 1000 x 5MB files (10 runs) ---")
    file_dir = await create_dir_with_files(num_files=1000, size_megabytes=5)

    print("Running directory downloads in parallel (each method runs 10 iterations)...")
    # Run the native tasks (they handle 10 iterations internally)
    td1 = asyncio.create_task(read_dir_new(file_dir, iterations=iterations))
    td2 = asyncio.create_task(read_dir_s3fs(file_dir, iterations=iterations))

    # Run s5cmd 10 times in parallel
    async def run_s5cmd_dir_multiple():
        print("[s5cmd Dir] Starting 10 parallel runs...", flush=True)
        tasks = [s5cmd_dir_task(remote_path=file_dir.path, expected_files=1000) for _ in range(iterations)]
        results = await asyncio.gather(*tasks)
        # Convert s5cmd results (duration, throughput) to BenchmarkResult
        dir_size_bytes = 1000 * 5 * 1024 * 1024
        return [BenchmarkResult(bytes=dir_size_bytes, duration=r[0]) for r in results]

    td3 = asyncio.create_task(run_s5cmd_dir_multiple())

    dir_results_new, dir_results_s3fs, dir_results_s5cmd = await asyncio.gather(td1, td2, td3)

    # Print summary
    avg_time_new_dir = sum(r.duration for r in dir_results_new) / len(dir_results_new)
    avg_time_s3fs_dir = sum(r.duration for r in dir_results_s3fs) / len(dir_results_s3fs)
    avg_time_s5cmd_dir = sum(r.duration for r in dir_results_s5cmd) / len(dir_results_s5cmd)

    print("\nDirectory average results:")
    print(f"  New SDK:  {avg_time_new_dir:.2f}s avg")
    print(f"  s3fs:     {avg_time_s3fs_dir:.2f}s avg")
    print(f"  s5cmd:    {avg_time_s5cmd_dir:.2f}s avg")

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)

    # Generate and publish report
    html = generate_benchmark_report(
        file_results_new=file_results_new,
        file_results_s3fs=file_results_s3fs,
        file_results_s5cmd=file_results_s5cmd,
        dir_results_new=dir_results_new,
        dir_results_s3fs=dir_results_s3fs,
        dir_results_s5cmd=dir_results_s5cmd,
    )

    await flyte.report.replace.aio(html)
    await flyte.report.flush.aio()


# From command line:
# $ flyte -c ~/.flyte/builder.remote.demo.yaml run -p flytesnacks -d development stress/benchmark/large_io_comparison.py benchmark_all  # noqa: E501
if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(benchmark_all)
    print(r.url)
