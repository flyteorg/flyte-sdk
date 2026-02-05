"""
Scale Test: Varied Container Images

Tests containerd snapshotter stability by launching N concurrent pods distributed
across 10 different container images with varying sizes. Each pod performs
comprehensive stress testing with library-specific patterns:
- Deep library imports to trigger lazy loading
- Filesystem traversal of site-packages
- Memory pressure with library-native operations
- Continuous periodic activity

This stresses:
- Snapshotter's lazy loading with multiple different images
- Multiple pods with different images running on the same node
- Content-addressed storage efficiency across varied image layers

Usage:
    uv run --active python examples/stress/scale_test_varied_images.py
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

# Memory and cycle settings per image type
HEAVY_CYCLE_SECONDS = 30  # PyTorch, TensorFlow, JAX, Transformers
MEDIUM_CYCLE_SECONDS = 40  # SciPy, Vision, Data
LIGHT_CYCLE_SECONDS = 60  # Requests, Minimal, Compute


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
    files_traversed: int = 0
    bytes_read: int = 0
    memory_cycles: int = 0


@dataclass
class ImageStats:
    """Statistics for a single image type."""

    success: int = 0
    failed: int = 0
    total_import_time: float = 0.0
    total_files_traversed: int = 0
    total_bytes_read: int = 0


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
# STRESS TEST UTILITIES
# ============================================================================


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
            depth = len(Path(root).relative_to(site_packages).parts)
            if depth > 4:
                dirs.clear()
                continue

            for filename in files[:50]:
                filepath = os.path.join(root, filename)
                try:
                    stat = os.stat(filepath)
                    total_size += stat.st_size
                    file_count += 1
                except (OSError, PermissionError):
                    pass

            if file_count > 5000:
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

    so_files = list(site_packages.rglob("*.so"))[:100]
    files_read = 0
    bytes_read = 0

    for so_file in so_files[:30]:
        try:
            size = so_file.stat().st_size
            if size < 1024:
                continue

            with open(so_file, "rb") as f:
                for _ in range(min(5, size // 4096)):
                    offset = random.randint(0, max(0, size - 4096))
                    f.seek(offset)
                    chunk = f.read(4096)
                    bytes_read += len(chunk)

            files_read += 1
        except Exception:
            pass

    return files_read, bytes_read


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
# TASK ENVIRONMENT DEFINITIONS
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
# WORKER TASKS
# ============================================================================


@pytorch_env.task
async def worker_pytorch(pod_id: int) -> WorkerResult:
    """PyTorch worker with deep imports and tensor memory pressure."""
    start = time.time()
    total_files = 0
    total_bytes = 0
    memory_cycles = 0

    # Phase 1: Deep imports
    print(f"[PyTorch Pod {pod_id}] Phase 1 - Deep imports...")
    import_start = time.time()

    import torch
    import torch.nn as nn

    _ = nn.Linear(10, 5)
    _ = torch.zeros(1)

    import_time = time.time() - import_start
    print(f"[PyTorch Pod {pod_id}] Imports complete in {import_time:.2f}s")

    # Phase 2: Filesystem stress
    files, size = stress_filesystem_traversal()
    total_files += files
    _, bytes_read = stress_file_content_reading()
    total_bytes += bytes_read

    # Phase 3: Stress loop
    duration = SLEEP_DURATION.total_seconds()
    num_cycles = int(duration / HEAVY_CYCLE_SECONDS)
    print(f"[PyTorch Pod {pod_id}] Starting {num_cycles} stress cycles...")

    for cycle in range(num_cycles):
        cycle_start = time.time()

        try:
            # Memory pressure with PyTorch tensors
            weights = [torch.randn(512, 512) for _ in range(8)]
            data = torch.randn(16, 128, 512)
            activations = [torch.matmul(data, w.T) for w in weights[:4]]
            grads = [torch.randn_like(w) for w in weights[:4]]
            time.sleep(0.3)
            del weights, data, activations, grads
            gc.collect()
            memory_cycles += 1
        except MemoryError:
            gc.collect()

        if cycle % 3 == 0:
            f, _ = stress_filesystem_traversal()
            total_files += f
        if cycle % 5 == 0:
            _, b = stress_file_content_reading()
            total_bytes += b

        elapsed = time.time() - cycle_start
        await asyncio.sleep(max(0, HEAVY_CYCLE_SECONDS - elapsed))

    total_elapsed = time.time() - start
    print(f"[PyTorch Pod {pod_id}] Complete in {total_elapsed:.2f}s")

    return WorkerResult(
        pod_id=pod_id,
        image_type="pytorch",
        status="success",
        import_time=round(import_time, 2),
        elapsed_seconds=round(total_elapsed, 2),
        files_traversed=total_files,
        bytes_read=total_bytes,
        memory_cycles=memory_cycles,
    )


@tensorflow_env.task
async def worker_tensorflow(pod_id: int) -> WorkerResult:
    """TensorFlow worker with deep imports and tensor memory pressure."""
    start = time.time()
    total_files = 0
    total_bytes = 0
    memory_cycles = 0

    # Phase 1: Deep imports
    print(f"[TensorFlow Pod {pod_id}] Phase 1 - Deep imports...")
    import_start = time.time()

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.config.set_visible_devices([], "GPU")
    _ = keras.Sequential([layers.Dense(10)])

    import_time = time.time() - import_start
    print(f"[TensorFlow Pod {pod_id}] Imports complete in {import_time:.2f}s")

    # Phase 2: Filesystem stress
    files, _ = stress_filesystem_traversal()
    total_files += files
    _, bytes_read = stress_file_content_reading()
    total_bytes += bytes_read

    # Phase 3: Stress loop
    duration = SLEEP_DURATION.total_seconds()
    num_cycles = int(duration / HEAVY_CYCLE_SECONDS)
    print(f"[TensorFlow Pod {pod_id}] Starting {num_cycles} stress cycles...")

    for cycle in range(num_cycles):
        cycle_start = time.time()

        try:
            # Memory pressure with TensorFlow tensors
            with tf.device("/CPU:0"):
                images = tf.zeros((16, 112, 112, 3), dtype=tf.float32)
                filters = [tf.zeros((3, 3, 3, 32), dtype=tf.float32) for _ in range(4)]
                features = [tf.nn.conv2d(images, f, strides=1, padding="SAME") for f in filters]
                time.sleep(0.3)
                del images, filters, features
            gc.collect()
            memory_cycles += 1
        except MemoryError:
            gc.collect()

        if cycle % 3 == 0:
            f, _ = stress_filesystem_traversal()
            total_files += f
        if cycle % 5 == 0:
            _, b = stress_file_content_reading()
            total_bytes += b

        elapsed = time.time() - cycle_start
        await asyncio.sleep(max(0, HEAVY_CYCLE_SECONDS - elapsed))

    total_elapsed = time.time() - start
    print(f"[TensorFlow Pod {pod_id}] Complete in {total_elapsed:.2f}s")

    return WorkerResult(
        pod_id=pod_id,
        image_type="tensorflow",
        status="success",
        import_time=round(import_time, 2),
        elapsed_seconds=round(total_elapsed, 2),
        files_traversed=total_files,
        bytes_read=total_bytes,
        memory_cycles=memory_cycles,
    )


@jax_env.task
async def worker_jax(pod_id: int) -> WorkerResult:
    """JAX worker with deep imports and array memory pressure."""
    start = time.time()
    total_files = 0
    total_bytes = 0
    memory_cycles = 0

    # Phase 1: Deep imports
    print(f"[JAX Pod {pod_id}] Phase 1 - Deep imports...")
    import_start = time.time()

    import jax
    import jax.numpy as jnp

    _ = jnp.array([1, 2, 3])

    import_time = time.time() - import_start
    print(f"[JAX Pod {pod_id}] Imports complete in {import_time:.2f}s")

    # Phase 2: Filesystem stress
    files, _ = stress_filesystem_traversal()
    total_files += files
    _, bytes_read = stress_file_content_reading()
    total_bytes += bytes_read

    # Phase 3: Stress loop
    duration = SLEEP_DURATION.total_seconds()
    num_cycles = int(duration / HEAVY_CYCLE_SECONDS)
    print(f"[JAX Pod {pod_id}] Starting {num_cycles} stress cycles...")

    for cycle in range(num_cycles):
        cycle_start = time.time()

        try:
            # Memory pressure with JAX arrays
            key = jax.random.PRNGKey(cycle)
            params = [jax.random.normal(key, (512, 512)) for _ in range(6)]
            batch = jax.random.normal(key, (32, 512))
            result = batch
            for p in params[:3]:
                result = jnp.dot(result, p)
            time.sleep(0.3)
            del params, batch, result
            gc.collect()
            memory_cycles += 1
        except MemoryError:
            gc.collect()

        if cycle % 3 == 0:
            f, _ = stress_filesystem_traversal()
            total_files += f
        if cycle % 5 == 0:
            _, b = stress_file_content_reading()
            total_bytes += b

        elapsed = time.time() - cycle_start
        await asyncio.sleep(max(0, HEAVY_CYCLE_SECONDS - elapsed))

    total_elapsed = time.time() - start
    print(f"[JAX Pod {pod_id}] Complete in {total_elapsed:.2f}s")

    return WorkerResult(
        pod_id=pod_id,
        image_type="jax",
        status="success",
        import_time=round(import_time, 2),
        elapsed_seconds=round(total_elapsed, 2),
        files_traversed=total_files,
        bytes_read=total_bytes,
        memory_cycles=memory_cycles,
    )


@transformers_env.task
async def worker_transformers(pod_id: int) -> WorkerResult:
    """Transformers worker with deep imports and tokenizer/embedding memory pressure."""
    start = time.time()
    total_files = 0
    total_bytes = 0
    memory_cycles = 0

    # Phase 1: Deep imports
    print(f"[Transformers Pod {pod_id}] Phase 1 - Deep imports...")
    import_start = time.time()

    import numpy as np
    import transformers

    _ = transformers.__version__

    import_time = time.time() - import_start
    print(f"[Transformers Pod {pod_id}] Imports complete in {import_time:.2f}s")

    # Phase 2: Filesystem stress
    files, _ = stress_filesystem_traversal()
    total_files += files
    _, bytes_read = stress_file_content_reading()
    total_bytes += bytes_read

    # Phase 3: Stress loop
    duration = SLEEP_DURATION.total_seconds()
    num_cycles = int(duration / HEAVY_CYCLE_SECONDS)
    print(f"[Transformers Pod {pod_id}] Starting {num_cycles} stress cycles...")

    for cycle in range(num_cycles):
        cycle_start = time.time()

        try:
            # Memory pressure simulating embedding/attention
            batch_size = 16
            seq_length = 256
            hidden_size = 384
            num_heads = 6

            embeddings = np.zeros((30000, hidden_size), dtype=np.float32)
            attention = [
                np.random.randn(batch_size, num_heads, seq_length, seq_length).astype(np.float32) * 0.01
                for _ in range(3)
            ]
            hidden = np.random.randn(batch_size, seq_length, hidden_size).astype(np.float32)
            time.sleep(0.3)
            del embeddings, attention, hidden
            gc.collect()
            memory_cycles += 1
        except MemoryError:
            gc.collect()

        if cycle % 3 == 0:
            f, _ = stress_filesystem_traversal()
            total_files += f
        if cycle % 5 == 0:
            _, b = stress_file_content_reading()
            total_bytes += b

        elapsed = time.time() - cycle_start
        await asyncio.sleep(max(0, HEAVY_CYCLE_SECONDS - elapsed))

    total_elapsed = time.time() - start
    print(f"[Transformers Pod {pod_id}] Complete in {total_elapsed:.2f}s")

    return WorkerResult(
        pod_id=pod_id,
        image_type="transformers",
        status="success",
        import_time=round(import_time, 2),
        elapsed_seconds=round(total_elapsed, 2),
        files_traversed=total_files,
        bytes_read=total_bytes,
        memory_cycles=memory_cycles,
    )


@scipy_env.task
async def worker_scipy(pod_id: int) -> WorkerResult:
    """SciPy worker with deep imports and scientific computing memory pressure."""
    start = time.time()
    total_files = 0
    total_bytes = 0
    memory_cycles = 0

    # Phase 1: Deep imports
    print(f"[SciPy Pod {pod_id}] Phase 1 - Deep imports...")
    import_start = time.time()

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    _ = RandomForestClassifier(n_estimators=1)

    import_time = time.time() - import_start
    print(f"[SciPy Pod {pod_id}] Imports complete in {import_time:.2f}s")

    # Phase 2: Filesystem stress
    files, _ = stress_filesystem_traversal()
    total_files += files
    _, bytes_read = stress_file_content_reading()
    total_bytes += bytes_read

    # Phase 3: Stress loop
    duration = SLEEP_DURATION.total_seconds()
    num_cycles = int(duration / MEDIUM_CYCLE_SECONDS)
    print(f"[SciPy Pod {pod_id}] Starting {num_cycles} stress cycles...")

    for cycle in range(num_cycles):
        cycle_start = time.time()

        try:
            # Memory pressure with numpy/scipy operations
            n_samples = 500000
            n_features = 20
            data = np.random.randn(n_samples, n_features).astype(np.float32)
            df = pd.DataFrame(data[:100000], columns=[f"f{i}" for i in range(n_features)])
            df_norm = (df - df.mean()) / df.std()
            cov = np.cov(data[:50000].T)
            eigenvalues = np.linalg.eigvals(cov)
            time.sleep(0.5)
            del data, df, df_norm, cov, eigenvalues
            gc.collect()
            memory_cycles += 1
        except MemoryError:
            gc.collect()

        if cycle % 3 == 0:
            f, _ = stress_filesystem_traversal()
            total_files += f
        if cycle % 5 == 0:
            _, b = stress_file_content_reading()
            total_bytes += b

        elapsed = time.time() - cycle_start
        await asyncio.sleep(max(0, MEDIUM_CYCLE_SECONDS - elapsed))

    total_elapsed = time.time() - start
    print(f"[SciPy Pod {pod_id}] Complete in {total_elapsed:.2f}s")

    return WorkerResult(
        pod_id=pod_id,
        image_type="scipy",
        status="success",
        import_time=round(import_time, 2),
        elapsed_seconds=round(total_elapsed, 2),
        files_traversed=total_files,
        bytes_read=total_bytes,
        memory_cycles=memory_cycles,
    )


@vision_env.task
async def worker_vision(pod_id: int) -> WorkerResult:
    """Vision worker with deep imports and image processing memory pressure."""
    start = time.time()
    total_files = 0
    total_bytes = 0
    memory_cycles = 0

    # Phase 1: Deep imports
    print(f"[Vision Pod {pod_id}] Phase 1 - Deep imports...")
    import_start = time.time()

    import cv2
    import numpy as np

    _ = cv2.__version__

    import_time = time.time() - import_start
    print(f"[Vision Pod {pod_id}] Imports complete in {import_time:.2f}s")

    # Phase 2: Filesystem stress
    files, _ = stress_filesystem_traversal()
    total_files += files
    _, bytes_read = stress_file_content_reading()
    total_bytes += bytes_read

    # Phase 3: Stress loop
    duration = SLEEP_DURATION.total_seconds()
    num_cycles = int(duration / MEDIUM_CYCLE_SECONDS)
    print(f"[Vision Pod {pod_id}] Starting {num_cycles} stress cycles...")

    for cycle in range(num_cycles):
        cycle_start = time.time()

        try:
            # Memory pressure with image arrays
            batch_size = 32
            images = np.random.randint(0, 255, size=(batch_size, 512, 512, 3), dtype=np.uint8)
            processed = []
            for i in range(min(batch_size, 16)):
                img = images[i]
                resized = cv2.resize(img, (256, 256))
                blurred = cv2.GaussianBlur(resized, (5, 5), 0)
                processed.append(blurred)
            features = [img.flatten()[:500] for img in processed[:8]]
            time.sleep(0.5)
            del images, processed, features
            gc.collect()
            memory_cycles += 1
        except MemoryError:
            gc.collect()

        if cycle % 3 == 0:
            f, _ = stress_filesystem_traversal()
            total_files += f
        if cycle % 5 == 0:
            _, b = stress_file_content_reading()
            total_bytes += b

        elapsed = time.time() - cycle_start
        await asyncio.sleep(max(0, MEDIUM_CYCLE_SECONDS - elapsed))

    total_elapsed = time.time() - start
    print(f"[Vision Pod {pod_id}] Complete in {total_elapsed:.2f}s")

    return WorkerResult(
        pod_id=pod_id,
        image_type="vision",
        status="success",
        import_time=round(import_time, 2),
        elapsed_seconds=round(total_elapsed, 2),
        files_traversed=total_files,
        bytes_read=total_bytes,
        memory_cycles=memory_cycles,
    )


@data_env.task
async def worker_data(pod_id: int) -> WorkerResult:
    """Data worker with deep imports and DataFrame memory pressure."""
    start = time.time()
    total_files = 0
    total_bytes = 0
    memory_cycles = 0

    # Phase 1: Deep imports
    print(f"[Data Pod {pod_id}] Phase 1 - Deep imports...")
    import_start = time.time()

    import polars as pl

    _ = pl.__version__

    import_time = time.time() - import_start
    print(f"[Data Pod {pod_id}] Imports complete in {import_time:.2f}s")

    # Phase 2: Filesystem stress
    files, _ = stress_filesystem_traversal()
    total_files += files
    _, bytes_read = stress_file_content_reading()
    total_bytes += bytes_read

    # Phase 3: Stress loop
    duration = SLEEP_DURATION.total_seconds()
    num_cycles = int(duration / MEDIUM_CYCLE_SECONDS)
    print(f"[Data Pod {pod_id}] Starting {num_cycles} stress cycles...")

    for cycle in range(num_cycles):
        cycle_start = time.time()

        try:
            # Memory pressure with DataFrames
            n_rows = 2000000
            df = pl.DataFrame(
                {
                    "id": range(n_rows),
                    "value1": [i * 0.1 for i in range(n_rows)],
                    "value2": [i * 0.2 for i in range(n_rows)],
                    "category": [f"cat_{i % 50}" for i in range(n_rows)],
                }
            )
            df_filtered = df.filter(pl.col("value1") > n_rows * 0.05)
            df_agg = df_filtered.group_by("category").agg(
                [pl.col("value1").mean().alias("mean_v1"), pl.col("value2").sum().alias("sum_v2")]
            )
            arrow_table = df_agg.to_arrow()
            time.sleep(0.5)
            del df, df_filtered, df_agg, arrow_table
            gc.collect()
            memory_cycles += 1
        except MemoryError:
            gc.collect()

        if cycle % 3 == 0:
            f, _ = stress_filesystem_traversal()
            total_files += f
        if cycle % 5 == 0:
            _, b = stress_file_content_reading()
            total_bytes += b

        elapsed = time.time() - cycle_start
        await asyncio.sleep(max(0, MEDIUM_CYCLE_SECONDS - elapsed))

    total_elapsed = time.time() - start
    print(f"[Data Pod {pod_id}] Complete in {total_elapsed:.2f}s")

    return WorkerResult(
        pod_id=pod_id,
        image_type="data",
        status="success",
        import_time=round(import_time, 2),
        elapsed_seconds=round(total_elapsed, 2),
        files_traversed=total_files,
        bytes_read=total_bytes,
        memory_cycles=memory_cycles,
    )


@requests_env.task
async def worker_requests(pod_id: int) -> WorkerResult:
    """Requests worker with imports and light memory pressure."""
    start = time.time()
    total_files = 0
    total_bytes = 0
    memory_cycles = 0

    # Phase 1: Imports
    print(f"[Requests Pod {pod_id}] Phase 1 - Imports...")
    import_start = time.time()

    import requests

    _ = requests.__version__

    import_time = time.time() - import_start
    print(f"[Requests Pod {pod_id}] Imports complete in {import_time:.2f}s")

    # Phase 2: Filesystem stress
    files, _ = stress_filesystem_traversal()
    total_files += files
    _, bytes_read = stress_file_content_reading()
    total_bytes += bytes_read

    # Phase 3: Stress loop (lighter pressure)
    duration = SLEEP_DURATION.total_seconds()
    num_cycles = int(duration / LIGHT_CYCLE_SECONDS)
    print(f"[Requests Pod {pod_id}] Starting {num_cycles} stress cycles...")

    for cycle in range(num_cycles):
        cycle_start = time.time()

        try:
            # Light memory pressure with response buffers
            mock_responses = []
            for i in range(50):
                response_data = {
                    "status": 200,
                    "headers": {"content-type": "application/json"},
                    "body": "x" * (400 * 1024),  # 400KB
                }
                mock_responses.append(response_data)
            parsed = [{"id": i, "data": list(range(500))} for i in range(500)]
            time.sleep(0.5)
            del mock_responses, parsed
            gc.collect()
            memory_cycles += 1
        except MemoryError:
            gc.collect()

        if cycle % 2 == 0:
            f, _ = stress_filesystem_traversal()
            total_files += f
        if cycle % 3 == 0:
            _, b = stress_file_content_reading()
            total_bytes += b

        elapsed = time.time() - cycle_start
        await asyncio.sleep(max(0, LIGHT_CYCLE_SECONDS - elapsed))

    total_elapsed = time.time() - start
    print(f"[Requests Pod {pod_id}] Complete in {total_elapsed:.2f}s")

    return WorkerResult(
        pod_id=pod_id,
        image_type="requests",
        status="success",
        import_time=round(import_time, 2),
        elapsed_seconds=round(total_elapsed, 2),
        files_traversed=total_files,
        bytes_read=total_bytes,
        memory_cycles=memory_cycles,
    )


@minimal_env.task
async def worker_minimal(pod_id: int) -> WorkerResult:
    """Minimal worker with light stress patterns."""
    start = time.time()
    total_files = 0
    total_bytes = 0
    memory_cycles = 0

    # Phase 1: Imports
    print(f"[Minimal Pod {pod_id}] Phase 1 - Imports...")
    import_start = time.time()

    import json

    import flyte as flyte_module

    _ = flyte_module.__version__

    import_time = time.time() - import_start
    print(f"[Minimal Pod {pod_id}] Imports complete in {import_time:.2f}s")

    # Phase 2: Filesystem stress
    files, _ = stress_filesystem_traversal()
    total_files += files
    _, bytes_read = stress_file_content_reading()
    total_bytes += bytes_read

    # Phase 3: Stress loop (lightest pressure)
    duration = SLEEP_DURATION.total_seconds()
    num_cycles = int(duration / LIGHT_CYCLE_SECONDS)
    print(f"[Minimal Pod {pod_id}] Starting {num_cycles} stress cycles...")

    for cycle in range(num_cycles):
        cycle_start = time.time()

        try:
            # Light memory pressure with data structures
            data = [{"id": i, "values": [j * 0.1 for j in range(50)], "meta": {"k": f"v{i}"}} for i in range(5000)]
            serialized = [json.dumps(d) for d in data[:500]]
            deserialized = [json.loads(s) for s in serialized]
            time.sleep(0.5)
            del data, serialized, deserialized
            gc.collect()
            memory_cycles += 1
        except MemoryError:
            gc.collect()

        if cycle % 2 == 0:
            f, _ = stress_filesystem_traversal()
            total_files += f
        if cycle % 3 == 0:
            _, b = stress_file_content_reading()
            total_bytes += b

        elapsed = time.time() - cycle_start
        await asyncio.sleep(max(0, LIGHT_CYCLE_SECONDS - elapsed))

    total_elapsed = time.time() - start
    print(f"[Minimal Pod {pod_id}] Complete in {total_elapsed:.2f}s")

    return WorkerResult(
        pod_id=pod_id,
        image_type="minimal",
        status="success",
        import_time=round(import_time, 2),
        elapsed_seconds=round(total_elapsed, 2),
        files_traversed=total_files,
        bytes_read=total_bytes,
        memory_cycles=memory_cycles,
    )


@compute_env.task
async def worker_compute(pod_id: int) -> WorkerResult:
    """Compute worker with numerical operations memory pressure."""
    start = time.time()
    total_files = 0
    total_bytes = 0
    memory_cycles = 0

    # Phase 1: Imports
    print(f"[Compute Pod {pod_id}] Phase 1 - Imports...")
    import_start = time.time()

    import numba
    import numpy as np

    _ = numba.__version__

    import_time = time.time() - import_start
    print(f"[Compute Pod {pod_id}] Imports complete in {import_time:.2f}s")

    # Phase 2: Filesystem stress
    files, _ = stress_filesystem_traversal()
    total_files += files
    _, bytes_read = stress_file_content_reading()
    total_bytes += bytes_read

    # Phase 3: Stress loop
    duration = SLEEP_DURATION.total_seconds()
    num_cycles = int(duration / LIGHT_CYCLE_SECONDS)
    print(f"[Compute Pod {pod_id}] Starting {num_cycles} stress cycles...")

    for cycle in range(num_cycles):
        cycle_start = time.time()

        try:
            # Memory pressure with numerical arrays
            n = 5000000
            arr1 = np.random.randn(n).astype(np.float32)
            arr2 = np.random.randn(n).astype(np.float32)
            result1 = arr1 * arr2
            result2 = np.sin(arr1) + np.cos(arr2)
            result3 = np.cumsum(result1[:500000])
            fft_result = np.fft.fft(arr1[:50000])
            time.sleep(0.5)
            del arr1, arr2, result1, result2, result3, fft_result
            gc.collect()
            memory_cycles += 1
        except MemoryError:
            gc.collect()

        if cycle % 2 == 0:
            f, _ = stress_filesystem_traversal()
            total_files += f
        if cycle % 3 == 0:
            _, b = stress_file_content_reading()
            total_bytes += b

        elapsed = time.time() - cycle_start
        await asyncio.sleep(max(0, LIGHT_CYCLE_SECONDS - elapsed))

    total_elapsed = time.time() - start
    print(f"[Compute Pod {pod_id}] Complete in {total_elapsed:.2f}s")

    return WorkerResult(
        pod_id=pod_id,
        image_type="compute",
        status="success",
        import_time=round(import_time, 2),
        elapsed_seconds=round(total_elapsed, 2),
        files_traversed=total_files,
        bytes_read=total_bytes,
        memory_cycles=memory_cycles,
    )


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

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
    """
    print("=" * 70)
    print(f"SCALE TEST: Launching {n} pods across 10 different images")
    print("Stress patterns: deep imports + filesystem + memory pressure")
    print("Images: pytorch, tensorflow, jax, transformers, scipy,")
    print("        vision, data, requests, minimal, compute")
    print("=" * 70)

    start_time = time.time()

    # Distribute pods across images
    num_images = len(WORKERS)
    pods_per_image = n // num_images
    remainder = n % num_images

    tasks = []
    for i in range(pods_per_image):
        for worker_fn in WORKERS:
            tasks.append(worker_fn(i))
    for i in range(remainder):
        tasks.append(WORKERS[i](pods_per_image))

    print(f"Launching {len(tasks)} concurrent worker tasks...")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed_time = time.time() - start_time

    # Analyze results
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
        stats.total_files_traversed += result.files_traversed
        stats.total_bytes_read += result.bytes_read

    for exc in failures:
        print(f"Task failed: {exc}")

    # Print report
    print("\n" + "=" * 70)
    print("SCALE TEST RESULTS:")
    print(f"  Total Pods:        {n}")
    print(f"  Successful:        {len(successes)}")
    print(f"  Failed:            {len(failures)}")
    print(f"  Success Rate:      {(len(successes) / n) * 100:.1f}%")
    print(f"  Total Elapsed:     {elapsed_time:.2f}s")
    print("=" * 70)
    print("\nResults by Image Type:")
    print(f"  {'Image':<15} {'Success':>8} {'Avg Import':>12} {'Files':>10} {'Bytes Read':>12}")
    print(f"  {'-' * 15} {'-' * 8} {'-' * 12} {'-' * 10} {'-' * 12}")
    for img_type in IMAGE_TYPES:
        stats = summary[img_type]
        avg_import = stats.total_import_time / stats.success if stats.success > 0 else 0
        bytes_mb = stats.total_bytes_read / 1e6
        print(
            f"  {img_type:<15} {stats.success:>8} {avg_import:>10.2f}s {stats.total_files_traversed:>10,} {bytes_mb:>10.1f}MB"
        )
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

    run = flyte.run(scale_test_orchestrator, n=10)

    print("\nScale Test Submitted:")
    print(f"  Run Name: {run.name}")
    print(f"  Run URL:  {run.url}")
    print(f"\nMonitor progress at: {run.url}")

    run.wait()
