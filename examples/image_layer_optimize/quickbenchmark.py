"""
Quick benchmark using scikit-learn instead of torch for faster results.

This benchmark uses a pre-built base image to avoid warming the cache inside the benchmark function.
"""

import asyncio
import logging
import time
from typing import Dict

import flyte
from flyte import Image
from flyte._internal.imagebuild.image_builder import ImageBuildEngine

# ============================================================================
# Create benchmark environment that uses the SAME base image
# ============================================================================
# By using the same Image definition, it will reuse the cached layers
benchmark_env = flyte.TaskEnvironment(
    name="benchmark",
    image=(Image.from_debian_base(name="benchmark-base").with_pip_packages("scikit-learn", "pandas")),
)


@benchmark_env.task
async def quick_benchmark() -> Dict[str, float]:
    """
    Quick benchmark using scikit-learn instead of torch for faster results.

    This assumes the base image is already built (cache is warm).
    """
    print("🔥 Quick Benchmark: Layer Optimization")

    # Phase 1: No optimization (rebuild all)
    print("\n[1/2] Adding 'requests' WITHOUT optimization...")
    no_opt = Image.from_debian_base(name="quick-no-opt").with_pip_packages("scikit-learn", "pandas", "requests")

    start = time.time()
    await ImageBuildEngine.build(no_opt, force=True, optimize_layers=False)
    no_opt_time = time.time() - start
    print(f"  ✓ Done in {no_opt_time:.1f}s")

    await asyncio.sleep(1)

    # Phase 2: With optimization (cache hit on scikit-learn)
    print("\n[2/2] Adding 'httpx' WITH optimization...")
    opt = Image.from_debian_base(name="quick-opt").with_pip_packages("scikit-learn", "pandas", "httpx")

    start = time.time()
    await ImageBuildEngine.build(opt, force=True, optimize_layers=True)
    opt_time = time.time() - start
    print(f"  ✓ Done in {opt_time:.1f}s")

    # Results
    speedup = no_opt_time / opt_time if opt_time > 0 else 1.0

    print("\n" + "=" * 60)
    print(f"Without optimization:   {no_opt_time:5.1f}s")
    print(f"With optimization:      {opt_time:5.1f}s")
    print(f"Speedup:                {speedup:5.1f}x")
    print("=" * 60)

    return {
        "no_opt_time": no_opt_time,
        "opt_time": opt_time,
        "speedup": speedup,
    }


if __name__ == "__main__":
    flyte.init_from_config(log_level=logging.DEBUG)
    run = flyte.with_runcontext(mode="remote", log_level=logging.DEBUG).run(quick_benchmark)
    print(run.name)
    print(run.url)
