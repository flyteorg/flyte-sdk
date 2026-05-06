"""
Scale Test: Same Container Image

Tests containerd snapshotter stability by launching N concurrent pods all using
the SAME heavy ML container image. Each pod performs comprehensive stress testing:
- Deep library imports to trigger lazy loading
- Filesystem traversal of site-packages
- Random content reading of .so files
- Memory pressure with sawtooth patterns

This stresses:
- Snapshotter's content-addressed storage efficiency
- Lazy loading performance under concurrent access
- Multiple pods with the same image running on the same node

Usage:
    uv run --active python examples/stress/scale_test_same_image.py
"""

import asyncio
import gc
import os
import random
import site
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import flyte

# ============================================================================
# CONSTANTS
# ============================================================================

SLEEP_DURATION = timedelta(minutes=5)

# Memory targets (stay well under 512Mi limit)
MEMORY_TARGET_MB = 300  # Peak target ~300Mi
MEMORY_CYCLE_SECONDS = 30  # Sawtooth cycle duration


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class WorkerResult:
    """Result from a single worker task."""

    worker_id: int
    status: str
    import_time: float
    elapsed_seconds: float
    files_traversed: int = 0
    bytes_read: int = 0
    memory_cycles: int = 0


@dataclass
class FailedWorker:
    """Information about a failed worker."""

    worker_id: int
    error: str


@dataclass
class ScaleTestReport:
    """Final report from the scale test orchestrator."""

    total_workers: int
    successes: int
    failures: int
    success_rate: float
    avg_import_time: float
    elapsed_seconds: float
    total_files_traversed: int = 0
    total_bytes_read: int = 0
    failed_workers: list[FailedWorker] = field(default_factory=list)


# ============================================================================
# STRESS TEST UTILITIES
# ============================================================================


def stress_deep_imports() -> float:
    """
    Import deep submodules to trigger extensive lazy loading.
    Returns time taken for imports.
    """
    start = time.time()

    # PyTorch deep imports
    # Matplotlib

    # NumPy submodules
    import numpy as np

    # Pandas
    # SciPy deep imports
    # Sklearn submodules
    import torch
    import torch.nn as nn

    # PIL
    from sklearn.ensemble import RandomForestClassifier

    # Trigger actual usage to ensure full loading
    _ = nn.Linear(10, 5)
    _ = torch.zeros(1)
    _ = np.zeros(1)
    _ = RandomForestClassifier(n_estimators=1)

    return time.time() - start


def stress_filesystem_traversal() -> tuple[int, int]:
    """
    Walk site-packages to trigger metadata operations.
    Returns (file_count, total_size_bytes).
    """
    try:
        site_packages = Path(site.getsitepackages()[0])
    except (IndexError, AttributeError):
        return 0, 0

    file_count = 0
    total_size = 0

    try:
        for root, dirs, files in os.walk(site_packages):
            # Limit traversal depth to avoid excessive time
            depth = len(Path(root).relative_to(site_packages).parts)
            if depth > 4:
                dirs.clear()
                continue

            for filename in files[:50]:  # Limit files per directory
                filepath = os.path.join(root, filename)
                try:
                    stat = os.stat(filepath)
                    total_size += stat.st_size
                    file_count += 1
                except (OSError, PermissionError):
                    pass

            if file_count > 5000:  # Cap total files
                break
    except Exception:
        pass

    return file_count, total_size


def stress_file_content_reading() -> tuple[int, int]:
    """
    Read random chunks from .so files to stress chunk fetching.
    Returns (files_read, bytes_read).
    """
    try:
        site_packages = Path(site.getsitepackages()[0])
    except (IndexError, AttributeError):
        return 0, 0

    # Find .so files (binary shared libraries)
    so_files = list(site_packages.rglob("*.so"))[:100]

    files_read = 0
    bytes_read = 0

    for so_file in so_files[:30]:  # Read from up to 30 files
        try:
            size = so_file.stat().st_size
            if size < 1024:
                continue

            with open(so_file, "rb") as f:
                # Read random 4KB chunks
                for _ in range(min(5, size // 4096)):
                    offset = random.randint(0, max(0, size - 4096))
                    f.seek(offset)
                    chunk = f.read(4096)
                    bytes_read += len(chunk)

            files_read += 1
        except Exception:
            pass

    return files_read, bytes_read


def stress_memory_pytorch(cycle: int) -> None:
    """
    Create PyTorch tensors to exert memory pressure.
    Uses sawtooth pattern: allocate -> hold -> release.
    """
    import torch

    # Allocate tensors (~200-250Mi total)
    batch_size = 32
    hidden_dim = 1024
    num_layers = 8

    # Model weights simulation
    weights = [torch.randn(hidden_dim, hidden_dim) for _ in range(num_layers)]

    # Batch data simulation
    batch_data = torch.randn(batch_size, 256, hidden_dim)

    # Forward pass simulation (creates intermediates)
    activations = []
    x = batch_data
    for i, w in enumerate(weights[:4]):
        x = torch.matmul(x, w.T)
        activations.append(x)

    # Gradient simulation
    gradients = [torch.randn_like(w) for w in weights[:4]]

    # Hold briefly
    time.sleep(0.5)

    # Cleanup
    del weights, batch_data, activations, gradients, x
    gc.collect()


# ============================================================================
# IMAGE AND ENVIRONMENT DEFINITIONS
# ============================================================================

# Heavy ML container image with ~1.5GB+ of dependencies
heavy_ml_image = flyte.Image.from_debian_base().with_pip_packages(
    "torch",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "pillow",
    "matplotlib",
)

# Worker environment with the heavy image
worker_env = flyte.TaskEnvironment(
    name="scale_test_same_image_worker",
    image=heavy_ml_image,
    resources=flyte.Resources(cpu="500m", memory="512Mi"),
)


# ============================================================================
# WORKER TASK
# ============================================================================


@worker_env.task
async def heavy_ml_worker(worker_id: int) -> WorkerResult:
    """
    Comprehensive stress test worker that:
    1. Deep imports to trigger lazy loading
    2. Traverses filesystem for metadata stress
    3. Reads .so file content for chunk fetching stress
    4. Memory pressure with sawtooth pattern
    5. Continuous periodic activity
    """
    start = time.time()
    total_files_traversed = 0
    total_bytes_read = 0
    memory_cycles = 0

    print(f"Worker {worker_id}: Phase 1 - Deep imports...")

    # Phase 1: Deep imports (triggers extensive lazy loading)
    import_time = stress_deep_imports()
    print(f"Worker {worker_id}: Imports complete in {import_time:.2f}s")

    # Phase 2: Initial filesystem stress
    print(f"Worker {worker_id}: Phase 2 - Filesystem traversal...")
    files_traversed, total_size = stress_filesystem_traversal()
    total_files_traversed += files_traversed
    print(f"Worker {worker_id}: Traversed {files_traversed} files ({total_size / 1e6:.1f} MB)")

    # Phase 3: Content reading stress
    print(f"Worker {worker_id}: Phase 3 - Reading .so file content...")
    files_read, bytes_read = stress_file_content_reading()
    total_bytes_read += bytes_read
    print(f"Worker {worker_id}: Read {bytes_read / 1e6:.2f} MB from {files_read} .so files")

    # Phase 4: Main stress loop with memory pressure and periodic filesystem activity
    duration_seconds = SLEEP_DURATION.total_seconds()
    cycle_duration = MEMORY_CYCLE_SECONDS
    num_cycles = int(duration_seconds / cycle_duration)

    print(f"Worker {worker_id}: Phase 4 - Starting {num_cycles} stress cycles...")

    for cycle in range(num_cycles):
        cycle_start = time.time()

        # Memory pressure (sawtooth pattern)
        try:
            stress_memory_pytorch(cycle)
            memory_cycles += 1
        except MemoryError:
            print(f"Worker {worker_id}: MemoryError in cycle {cycle}, reducing pressure")
            gc.collect()

        # Periodic filesystem activity (every 3rd cycle)
        if cycle % 3 == 0:
            files_traversed, _ = stress_filesystem_traversal()
            total_files_traversed += files_traversed

        # Periodic content reading (every 5th cycle)
        if cycle % 5 == 0:
            _, bytes_read = stress_file_content_reading()
            total_bytes_read += bytes_read

        # Sleep remainder of cycle
        elapsed = time.time() - cycle_start
        sleep_time = max(0, cycle_duration - elapsed)
        await asyncio.sleep(sleep_time)

        if (cycle + 1) % 5 == 0:
            print(f"Worker {worker_id}: Cycle {cycle + 1}/{num_cycles} complete")

    elapsed = time.time() - start
    print(f"Worker {worker_id}: Complete in {elapsed:.2f}s")

    return WorkerResult(
        worker_id=worker_id,
        status="success",
        import_time=round(import_time, 2),
        elapsed_seconds=round(elapsed, 2),
        files_traversed=total_files_traversed,
        bytes_read=total_bytes_read,
        memory_cycles=memory_cycles,
    )


# ============================================================================
# ORCHESTRATOR
# ============================================================================

orchestrator_env = flyte.TaskEnvironment(
    name="scale_test_same_image_orchestrator",
    resources=flyte.Resources(cpu="500m", memory="512Mi"),
    depends_on=[worker_env],
)


@orchestrator_env.task
async def scale_test_orchestrator(n: int = 100) -> ScaleTestReport:
    """
    Orchestrates N concurrent worker tasks, all using the same heavy ML image.
    """
    print("=" * 70)
    print(f"SCALE TEST: Launching {n} concurrent workers")
    print("All workers use the SAME heavy ML container image")
    print("Stress patterns: deep imports + filesystem + memory pressure")
    print("=" * 70)

    start_time = time.time()

    # Create all worker coroutines
    coros = [heavy_ml_worker(i) for i in range(n)]

    # Execute concurrently with failure tolerance
    print(f"Executing {n} workers concurrently...")
    results = await asyncio.gather(*coros, return_exceptions=True)

    elapsed_time = time.time() - start_time

    # Analyze results
    successes: list[WorkerResult] = []
    failed_workers: list[FailedWorker] = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            failed_workers.append(FailedWorker(worker_id=i, error=str(result)))
            print(f"Worker {i} FAILED: {result}")
        else:
            successes.append(result)

    success_count = len(successes)
    failure_count = len(failed_workers)
    success_rate = (success_count / n) * 100 if n > 0 else 0

    # Calculate aggregates
    avg_import_time = 0.0
    total_files = 0
    total_bytes = 0
    if successes:
        avg_import_time = sum(r.import_time for r in successes) / len(successes)
        total_files = sum(r.files_traversed for r in successes)
        total_bytes = sum(r.bytes_read for r in successes)

    # Print summary
    print("\n" + "=" * 70)
    print("SCALE TEST RESULTS:")
    print(f"  Total Workers:      {n}")
    print(f"  Successes:          {success_count}")
    print(f"  Failures:           {failure_count}")
    print(f"  Success Rate:       {success_rate:.1f}%")
    print(f"  Avg Import Time:    {avg_import_time:.2f}s")
    print(f"  Total Files Traversed: {total_files:,}")
    print(f"  Total Bytes Read:   {total_bytes / 1e9:.2f} GB")
    print(f"  Total Elapsed:      {elapsed_time:.2f}s")
    if failed_workers:
        print(f"  Failed Workers:     {[f.worker_id for f in failed_workers[:10]]}")
        if len(failed_workers) > 10:
            print(f"                      ... and {len(failed_workers) - 10} more")
    print("=" * 70)

    return ScaleTestReport(
        total_workers=n,
        successes=success_count,
        failures=failure_count,
        success_rate=success_rate,
        avg_import_time=round(avg_import_time, 2),
        elapsed_seconds=round(elapsed_time, 2),
        total_files_traversed=total_files,
        total_bytes_read=total_bytes,
        failed_workers=failed_workers,
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.run(scale_test_orchestrator, n=10)

    print("\nScale Test Submitted:")
    print(f"  Run Name: {run.name}")
    print(f"  Run URL:  {run.url}")
    print(f"\nMonitor progress at: {run.url}")

    run.wait()
