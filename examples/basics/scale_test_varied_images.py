"""
Scale Test: Varied Container Images

Tests containerd snapshotter stability by launching N concurrent pods distributed
across 10 different container images with varying sizes, from heavy ML packages
(~2GB) to lightweight utilities (~50MB).

This stresses:
- Snapshotter's lazy loading with multiple different images
- Multiple pods with different images running on the same node
- Content-addressed storage efficiency across varied image layers

Usage:
    uv run --active python examples/basics/scale_test_varied_images.py
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

    pod_id: int
    image_type: str
    status: str
    import_time: float
    elapsed_seconds: float


@dataclass
class ImageStats:
    """Statistics for a single image type."""

    success: int = 0
    failed: int = 0
    total_import_time: float = 0.0


@dataclass
class ScaleTestReport:
    """Final report from the scale test orchestrator."""

    total_pods: int
    successful: int
    failed: int
    success_rate: float
    elapsed_seconds: float
    summary_by_image: dict[str, ImageStats] = field(default_factory=dict)


# ============================================================================
# IMAGE DEFINITIONS - 10 images with varying sizes
# ============================================================================

# Heavy Images (~1-2GB)
pytorch_image = flyte.Image.from_debian_base().with_pip_packages(
    "torch",
    "torchvision",
)

tensorflow_image = flyte.Image.from_debian_base().with_pip_packages(
    "tensorflow",
)

jax_image = flyte.Image.from_debian_base().with_pip_packages(
    "jax[cpu]",
)

transformers_image = flyte.Image.from_debian_base().with_pip_packages(
    "transformers",
    "tokenizers",
    "datasets",
)

# Medium Images (~400-800MB)
scipy_image = flyte.Image.from_debian_base().with_pip_packages(
    "scipy",
    "scikit-learn",
    "pandas",
    "numpy",
)

vision_image = (
    flyte.Image.from_debian_base()
    .with_apt_packages("libgl1-mesa-glx", "libglib2.0-0")
    .with_pip_packages(
        "opencv-python-headless",
        "pillow",
        "scikit-image",
    )
)

data_image = flyte.Image.from_debian_base().with_pip_packages(
    "polars",
    "pyarrow",
    "duckdb",
)

# Light Images (~50-200MB)
requests_image = flyte.Image.from_debian_base().with_pip_packages(
    "requests",
    "httpx",
    "aiohttp",
)

minimal_image = flyte.Image.from_debian_base().with_pip_packages(
    "flyte",
)

compute_image = flyte.Image.from_debian_base().with_pip_packages(
    "numba",
    "joblib",
    "numpy",
)

# ============================================================================
# TASK ENVIRONMENT DEFINITIONS - One per image
# ============================================================================

pytorch_env = flyte.TaskEnvironment(
    name="scale_test_pytorch",
    image=pytorch_image,
    resources=flyte.Resources(cpu="500m", memory="512Mi"),
)

tensorflow_env = flyte.TaskEnvironment(
    name="scale_test_tensorflow",
    image=tensorflow_image,
    resources=flyte.Resources(cpu="500m", memory="512Mi"),
)

jax_env = flyte.TaskEnvironment(
    name="scale_test_jax",
    image=jax_image,
    resources=flyte.Resources(cpu="500m", memory="512Mi"),
)

transformers_env = flyte.TaskEnvironment(
    name="scale_test_transformers",
    image=transformers_image,
    resources=flyte.Resources(cpu="500m", memory="512Mi"),
)

scipy_env = flyte.TaskEnvironment(
    name="scale_test_scipy",
    image=scipy_image,
    resources=flyte.Resources(cpu="500m", memory="512Mi"),
)

vision_env = flyte.TaskEnvironment(
    name="scale_test_vision",
    image=vision_image,
    resources=flyte.Resources(cpu="500m", memory="512Mi"),
)

data_env = flyte.TaskEnvironment(
    name="scale_test_data",
    image=data_image,
    resources=flyte.Resources(cpu="500m", memory="512Mi"),
)

requests_env = flyte.TaskEnvironment(
    name="scale_test_requests",
    image=requests_image,
    resources=flyte.Resources(cpu="500m", memory="512Mi"),
)

minimal_env = flyte.TaskEnvironment(
    name="scale_test_minimal",
    image=minimal_image,
    resources=flyte.Resources(cpu="500m", memory="512Mi"),
)

compute_env = flyte.TaskEnvironment(
    name="scale_test_compute",
    image=compute_image,
    resources=flyte.Resources(cpu="500m", memory="512Mi"),
)

# ============================================================================
# WORKER TASKS - One per environment
# ============================================================================


@pytorch_env.task
async def worker_pytorch(pod_id: int) -> WorkerResult:
    """Worker that imports PyTorch libraries to stress lazy loading."""
    start = time.time()

    import torch
    import torchvision

    _ = torch.__version__
    _ = torchvision.__version__

    import_time = time.time() - start
    sleep_seconds = SLEEP_DURATION.total_seconds()
    print(f"[PyTorch Pod {pod_id}] Imported in {import_time:.2f}s, sleeping {sleep_seconds:.0f}s...")
    await asyncio.sleep(sleep_seconds)

    elapsed = time.time() - start
    return WorkerResult(
        pod_id=pod_id,
        image_type="pytorch",
        status="success",
        import_time=round(import_time, 2),
        elapsed_seconds=round(elapsed, 2),
    )


@tensorflow_env.task
async def worker_tensorflow(pod_id: int) -> WorkerResult:
    """Worker that imports TensorFlow libraries to stress lazy loading."""
    start = time.time()

    import tensorflow as tf

    _ = tf.__version__

    import_time = time.time() - start
    sleep_seconds = SLEEP_DURATION.total_seconds()
    print(f"[TensorFlow Pod {pod_id}] Imported in {import_time:.2f}s, sleeping {sleep_seconds:.0f}s...")
    await asyncio.sleep(sleep_seconds)

    elapsed = time.time() - start
    return WorkerResult(
        pod_id=pod_id,
        image_type="tensorflow",
        status="success",
        import_time=round(import_time, 2),
        elapsed_seconds=round(elapsed, 2),
    )


@jax_env.task
async def worker_jax(pod_id: int) -> WorkerResult:
    """Worker that imports JAX libraries to stress lazy loading."""
    start = time.time()

    import jax
    import jax.numpy as jnp

    _ = jax.__version__
    _ = jnp.array([1, 2, 3])

    import_time = time.time() - start
    sleep_seconds = SLEEP_DURATION.total_seconds()
    print(f"[JAX Pod {pod_id}] Imported in {import_time:.2f}s, sleeping {sleep_seconds:.0f}s...")
    await asyncio.sleep(sleep_seconds)

    elapsed = time.time() - start
    return WorkerResult(
        pod_id=pod_id,
        image_type="jax",
        status="success",
        import_time=round(import_time, 2),
        elapsed_seconds=round(elapsed, 2),
    )


@transformers_env.task
async def worker_transformers(pod_id: int) -> WorkerResult:
    """Worker that imports Transformers libraries to stress lazy loading."""
    start = time.time()

    import datasets
    import tokenizers
    import transformers

    _ = transformers.__version__
    _ = tokenizers.__version__
    _ = datasets.__version__

    import_time = time.time() - start
    sleep_seconds = SLEEP_DURATION.total_seconds()
    print(f"[Transformers Pod {pod_id}] Imported in {import_time:.2f}s, sleeping {sleep_seconds:.0f}s...")
    await asyncio.sleep(sleep_seconds)

    elapsed = time.time() - start
    return WorkerResult(
        pod_id=pod_id,
        image_type="transformers",
        status="success",
        import_time=round(import_time, 2),
        elapsed_seconds=round(elapsed, 2),
    )


@scipy_env.task
async def worker_scipy(pod_id: int) -> WorkerResult:
    """Worker that imports SciPy stack to stress lazy loading."""
    start = time.time()

    import numpy as np
    import pandas as pd
    import scipy
    import sklearn

    _ = scipy.__version__
    _ = sklearn.__version__
    _ = pd.__version__
    _ = np.__version__

    import_time = time.time() - start
    sleep_seconds = SLEEP_DURATION.total_seconds()
    print(f"[SciPy Pod {pod_id}] Imported in {import_time:.2f}s, sleeping {sleep_seconds:.0f}s...")
    await asyncio.sleep(sleep_seconds)

    elapsed = time.time() - start
    return WorkerResult(
        pod_id=pod_id,
        image_type="scipy",
        status="success",
        import_time=round(import_time, 2),
        elapsed_seconds=round(elapsed, 2),
    )


@vision_env.task
async def worker_vision(pod_id: int) -> WorkerResult:
    """Worker that imports vision libraries to stress lazy loading."""
    start = time.time()

    import cv2
    import skimage
    from PIL import Image as PILImage

    _ = cv2.__version__
    _ = PILImage.__version__
    _ = skimage.__version__

    import_time = time.time() - start
    sleep_seconds = SLEEP_DURATION.total_seconds()
    print(f"[Vision Pod {pod_id}] Imported in {import_time:.2f}s, sleeping {sleep_seconds:.0f}s...")
    await asyncio.sleep(sleep_seconds)

    elapsed = time.time() - start
    return WorkerResult(
        pod_id=pod_id,
        image_type="vision",
        status="success",
        import_time=round(import_time, 2),
        elapsed_seconds=round(elapsed, 2),
    )


@data_env.task
async def worker_data(pod_id: int) -> WorkerResult:
    """Worker that imports data processing libraries to stress lazy loading."""
    start = time.time()

    import duckdb
    import polars as pl
    import pyarrow

    _ = pl.__version__
    _ = pyarrow.__version__
    _ = duckdb.__version__

    import_time = time.time() - start
    sleep_seconds = SLEEP_DURATION.total_seconds()
    print(f"[Data Pod {pod_id}] Imported in {import_time:.2f}s, sleeping {sleep_seconds:.0f}s...")
    await asyncio.sleep(sleep_seconds)

    elapsed = time.time() - start
    return WorkerResult(
        pod_id=pod_id,
        image_type="data",
        status="success",
        import_time=round(import_time, 2),
        elapsed_seconds=round(elapsed, 2),
    )


@requests_env.task
async def worker_requests(pod_id: int) -> WorkerResult:
    """Worker that imports lightweight HTTP libraries."""
    start = time.time()

    import aiohttp
    import httpx
    import requests

    _ = requests.__version__
    _ = httpx.__version__
    _ = aiohttp.__version__

    import_time = time.time() - start
    sleep_seconds = SLEEP_DURATION.total_seconds()
    print(f"[Requests Pod {pod_id}] Imported in {import_time:.2f}s, sleeping {sleep_seconds:.0f}s...")
    await asyncio.sleep(sleep_seconds)

    elapsed = time.time() - start
    return WorkerResult(
        pod_id=pod_id,
        image_type="requests",
        status="success",
        import_time=round(import_time, 2),
        elapsed_seconds=round(elapsed, 2),
    )


@minimal_env.task
async def worker_minimal(pod_id: int) -> WorkerResult:
    """Worker with minimal dependencies (just flyte)."""
    start = time.time()

    import flyte as flyte_module

    _ = flyte_module.__version__

    import_time = time.time() - start
    sleep_seconds = SLEEP_DURATION.total_seconds()
    print(f"[Minimal Pod {pod_id}] Imported in {import_time:.2f}s, sleeping {sleep_seconds:.0f}s...")
    await asyncio.sleep(sleep_seconds)

    elapsed = time.time() - start
    return WorkerResult(
        pod_id=pod_id,
        image_type="minimal",
        status="success",
        import_time=round(import_time, 2),
        elapsed_seconds=round(elapsed, 2),
    )


@compute_env.task
async def worker_compute(pod_id: int) -> WorkerResult:
    """Worker that imports lightweight compute libraries."""
    start = time.time()

    import joblib
    import numba
    import numpy as np

    _ = numba.__version__
    _ = joblib.__version__
    _ = np.__version__

    import_time = time.time() - start
    sleep_seconds = SLEEP_DURATION.total_seconds()
    print(f"[Compute Pod {pod_id}] Imported in {import_time:.2f}s, sleeping {sleep_seconds:.0f}s...")
    await asyncio.sleep(sleep_seconds)

    elapsed = time.time() - start
    return WorkerResult(
        pod_id=pod_id,
        image_type="compute",
        status="success",
        import_time=round(import_time, 2),
        elapsed_seconds=round(elapsed, 2),
    )


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

# List of all worker functions for distribution
WORKERS = [
    worker_pytorch,
    worker_tensorflow,
    worker_jax,
    worker_transformers,
    worker_scipy,
    worker_vision,
    worker_data,
    worker_requests,
    worker_minimal,
    worker_compute,
]

IMAGE_TYPES = [
    "pytorch",
    "tensorflow",
    "jax",
    "transformers",
    "scipy",
    "vision",
    "data",
    "requests",
    "minimal",
    "compute",
]

# Orchestrator environment - minimal resources
# depends_on ensures all worker images are built before orchestrator runs
orchestrator_env = flyte.TaskEnvironment(
    name="scale_test_varied_orchestrator",
    resources=flyte.Resources(cpu="500m", memory="512Mi"),
    depends_on=[
        pytorch_env,
        tensorflow_env,
        jax_env,
        transformers_env,
        scipy_env,
        vision_env,
        data_env,
        requests_env,
        minimal_env,
        compute_env,
    ],
)


@orchestrator_env.task
async def scale_test_orchestrator(n: int = 100) -> ScaleTestReport:
    """
    Main orchestrator that launches N concurrent pods distributed across 10 images.

    Args:
        n: Total number of pods to launch (default 100)

    Returns:
        ScaleTestReport with results per image type
    """
    print("=" * 70)
    print(f"SCALE TEST: Launching {n} pods across 10 different images")
    print("Images: pytorch, tensorflow, jax, transformers, scipy,")
    print("        vision, data, requests, minimal, compute")
    print("=" * 70)

    start_time = time.time()

    # Calculate pods per image (distribute evenly)
    num_images = len(WORKERS)
    pods_per_image = n // num_images
    remainder = n % num_images

    # Build list of all worker coroutines
    tasks = []
    for i in range(pods_per_image):
        for worker_fn in WORKERS:
            tasks.append(worker_fn(i))

    # Add remainder pods to first few workers
    for i in range(remainder):
        tasks.append(WORKERS[i](pods_per_image))

    print(f"Launching {len(tasks)} concurrent worker tasks...")

    # Launch all tasks concurrently with failure tolerance
    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed_time = time.time() - start_time

    # Separate successes and failures
    successes: list[WorkerResult] = []
    failures: list[Exception] = []
    for result in results:
        if isinstance(result, Exception):
            failures.append(result)
        else:
            successes.append(result)

    # Summarize by image type
    summary: dict[str, ImageStats] = {}
    for img_type in IMAGE_TYPES:
        summary[img_type] = ImageStats()

    for result in successes:
        stats = summary[result.image_type]
        stats.success += 1
        stats.total_import_time += result.import_time

    # Log failures
    for i, exc in enumerate(failures):
        print(f"Task {i} failed: {exc}")

    # Build final report
    print("\n" + "=" * 70)
    print("SCALE TEST RESULTS:")
    print(f"  Total Pods:        {n}")
    print(f"  Successful:        {len(successes)}")
    print(f"  Failed:            {len(failures)}")
    print(f"  Success Rate:      {(len(successes) / n) * 100:.1f}%")
    print(f"  Total Elapsed:     {elapsed_time:.2f}s")
    print("=" * 70)
    print("\nResults by Image Type:")
    print(f"  {'Image':<15} {'Success':>8} {'Avg Import':>12}")
    print(f"  {'-' * 15} {'-' * 8} {'-' * 12}")
    for img_type in IMAGE_TYPES:
        stats = summary[img_type]
        avg_import = stats.total_import_time / stats.success if stats.success > 0 else 0
        print(f"  {img_type:<15} {stats.success:>8} {avg_import:>10.2f}s")
    print("=" * 70)

    return ScaleTestReport(
        total_pods=n,
        successful=len(successes),
        failed=len(failures),
        success_rate=(len(successes) / n) * 100 if n > 0 else 0,
        elapsed_seconds=round(elapsed_time, 2),
        summary_by_image=summary,
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    flyte.init_from_config()

    # Run with default n=100 (10 pods per image)
    # Modify n parameter to test different scales
    run = flyte.run(scale_test_orchestrator, n=100)

    print("\nScale Test Submitted:")
    print(f"  Run Name: {run.name}")
    print(f"  Run URL:  {run.url}")
    print(f"\nMonitor progress at: {run.url}")

    run.wait()
