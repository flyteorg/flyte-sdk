"""
Benchmark: Docker layer optimization cache efficiency test.

This benchmark uses HEAVY dependencies (torch, tensorflow, transformers) to demonstrate
significant time savings from layer optimization.

Expected results:
- Without optimization: ~5-8 minutes (reinstalls ALL heavy packages)
- With optimization: ~10-30 seconds (reuses heavy layer cache)
- Speedup: 10-30x faster
"""

import logging
import time
from typing import Dict

import flyte
from flyte import Image
from flyte._internal.imagebuild.image_builder import ImageBuildEngine

# ============================================================================
# Base image with HEAVY dependencies - this warms the Docker cache
# ============================================================================
env = flyte.TaskEnvironment(
    name="benchmark",
    image=(
        Image.from_debian_base(name="benchmark-base").with_pip_packages(
            "torch",  # ~800MB
            "tensorflow",  # ~500MB
            "transformers",  # large w/ deps
            "numpy",
            "pandas",
        )
    ),
)


@env.task
async def benchmark_layer_optimization() -> Dict[str, float]:
    """
    Benchmark layer optimization with heavy ML dependencies.

    Phase 1: Add a small package WITHOUT optimization
    Phase 2: Add a small package WITH optimization
    """
    bar = "=" * 72
    print(bar)
    print("Docker Layer Optimization Benchmark (heavy deps)")
    print(bar)

    # ------------------------------------------------------------------------
    # Phase 1: WITHOUT optimization
    # ------------------------------------------------------------------------
    print("\n[1/2] WITHOUT optimization: add 'requests' (expect full rebuild)")
    image_no_opt = Image.from_debian_base(name="benchmark-no-opt").with_pip_packages(
        "torch",
        "tensorflow",
        "transformers",
        "numpy",
        "pandas",
        "requests",
    )

    start = time.time()
    await ImageBuildEngine.build(image_no_opt, force=True, optimize_layers=False)
    no_opt_time = time.time() - start
    print(f"  done: {no_opt_time:.1f}s  ({no_opt_time / 60:.1f} min)")

    # ------------------------------------------------------------------------
    # Phase 2: WITH optimization
    # ------------------------------------------------------------------------
    print("\n[2/2] WITH optimization: add 'httpx' (expect cache hit on heavy layer)")
    image_opt = Image.from_debian_base(name="benchmark-opt").with_pip_packages(
        "torch",
        "tensorflow",
        "transformers",
        "numpy",
        "pandas",
        "httpx",
    )

    start = time.time()
    await ImageBuildEngine.build(image_opt, force=True, optimize_layers=True)
    opt_time = time.time() - start
    print(f"  done: {opt_time:.1f}s  ({opt_time / 60:.1f} min)")

    # ------------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------------
    speedup = no_opt_time / opt_time if opt_time > 0 else 1.0
    time_saved = no_opt_time - opt_time

    print("\n" + bar)
    print("RESULTS")
    print(bar)
    print(f"no-opt:  {no_opt_time:7.1f}s  ({no_opt_time / 60:5.1f} min)  full rebuild")
    print(f"opt:     {opt_time:7.1f}s  ({opt_time / 60:5.1f} min)  cache reuse")
    print(f"speedup: {speedup:7.1f}x")
    print(f"saved:   {time_saved:7.1f}s  ({time_saved / 60:5.1f} min)")
    print(bar)

    return {
        "no_opt_time_seconds": no_opt_time,
        "no_opt_time_minutes": no_opt_time / 60,
        "opt_time_seconds": opt_time,
        "opt_time_minutes": opt_time / 60,
        "speedup": speedup,
        "time_saved_seconds": time_saved,
        "time_saved_minutes": time_saved / 60,
    }


if __name__ == "__main__":
    flyte.init_from_config(log_level=logging.DEBUG)
    run = flyte.with_runcontext(mode="remote", log_level=logging.DEBUG).run(benchmark_layer_optimization)
    print(run.name)
    print(run.url)
