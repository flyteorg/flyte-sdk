"""
Scale Test: Same Container Image

Tests containerd snapshotter stability by launching N concurrent pods all using
the SAME heavy ML container image. Each pod imports heavy libraries (PyTorch,
numpy, scipy, etc.) to trigger filesystem lazy loading, then sleeps and exits.

This stresses:
- Snapshotter's content-addressed storage efficiency
- Lazy loading performance under concurrent access
- Multiple pods with the same image running on the same node

Usage:
    uv run --active python examples/basics/scale_test_same_image.py
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import timedelta

import flyte

# ============================================================================
# CONSTANTS
# ============================================================================

SLEEP_DURATION = timedelta(minutes=5)


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
    failed_workers: list[FailedWorker] = field(default_factory=list)


# ============================================================================
# IMAGE AND ENVIRONMENT DEFINITIONS
# ============================================================================

# Heavy ML container image with ~1.5GB+ of dependencies
# This maximizes filesystem stress through numerous .so files and Python modules
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
    Worker task that imports heavy ML libraries and sleeps.

    This stresses the snapshotter's lazy loading by triggering imports
    of numerous large .so files and Python modules from the container image.

    Args:
        worker_id: Unique identifier for this worker

    Returns:
        WorkerResult with timing and status information
    """
    start = time.time()
    print(f"Worker {worker_id}: Starting imports...")

    # Import heavy libraries - each triggers lazy loading from snapshotter
    import matplotlib
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import torch
    from PIL import Image as PILImage

    # Do minimal work to ensure imports are used (prevents optimization)
    _ = torch.__version__
    _ = torch.zeros(1)
    _ = np.zeros(1)
    _ = scipy.__version__
    _ = pd.DataFrame({"a": [1, 2, 3]})
    _ = sklearn.__version__
    _ = matplotlib.__version__
    _ = PILImage.__version__

    import_time = time.time() - start
    sleep_seconds = SLEEP_DURATION.total_seconds()
    print(f"Worker {worker_id}: All imports complete in {import_time:.2f}s, sleeping {sleep_seconds:.0f}s...")

    await asyncio.sleep(sleep_seconds)

    elapsed = time.time() - start
    print(f"Worker {worker_id}: Complete in {elapsed:.2f}s")

    return WorkerResult(
        worker_id=worker_id,
        status="success",
        import_time=round(import_time, 2),
        elapsed_seconds=round(elapsed, 2),
    )


# ============================================================================
# ORCHESTRATOR
# ============================================================================

# Orchestrator environment - minimal resources
# depends_on ensures worker image is built before orchestrator runs
orchestrator_env = flyte.TaskEnvironment(
    name="scale_test_same_image_orchestrator",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    depends_on=[worker_env],
)


@orchestrator_env.task
async def scale_test_orchestrator(n: int = 100) -> ScaleTestReport:
    """
    Orchestrates N concurrent worker tasks, all using the same heavy ML image.

    Args:
        n: Number of concurrent workers to launch (default: 100)

    Returns:
        ScaleTestReport with test results including success/failure counts
    """
    print("=" * 70)
    print(f"SCALE TEST: Launching {n} concurrent workers")
    print("All workers use the SAME heavy ML container image")
    print("=" * 70)

    start_time = time.time()

    # Create all worker coroutines
    coros = [heavy_ml_worker(i) for i in range(n)]

    # Execute concurrently with failure tolerance
    # return_exceptions=True ensures we continue even if some workers fail
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

    # Calculate average import time for successful workers
    avg_import_time = 0.0
    if successes:
        avg_import_time = sum(r.import_time for r in successes) / len(successes)

    # Print summary
    print("\n" + "=" * 70)
    print("SCALE TEST RESULTS:")
    print(f"  Total Workers:     {n}")
    print(f"  Successes:         {success_count}")
    print(f"  Failures:          {failure_count}")
    print(f"  Success Rate:      {success_rate:.1f}%")
    print(f"  Avg Import Time:   {avg_import_time:.2f}s")
    print(f"  Total Elapsed:     {elapsed_time:.2f}s")
    if failed_workers:
        print(f"  Failed Workers:    {[f.worker_id for f in failed_workers[:10]]}")
        if len(failed_workers) > 10:
            print(f"                     ... and {len(failed_workers) - 10} more")
    print("=" * 70)

    return ScaleTestReport(
        total_workers=n,
        successes=success_count,
        failures=failure_count,
        success_rate=success_rate,
        avg_import_time=round(avg_import_time, 2),
        elapsed_seconds=round(elapsed_time, 2),
        failed_workers=failed_workers,
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    flyte.init_from_config()

    # Run scale test with default 100 workers
    # Modify n parameter to test different scales
    run = flyte.run(scale_test_orchestrator, n=100)

    print(f"\nScale Test Submitted:")
    print(f"  Run Name: {run.name}")
    print(f"  Run URL:  {run.url}")
    print(f"\nMonitor progress at: {run.url}")

    run.wait()
