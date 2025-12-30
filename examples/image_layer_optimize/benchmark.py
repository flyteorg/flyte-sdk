"""
Benchmark: Docker layer optimization cache efficiency test.
"""

import logging
import time
from typing import Dict

import flyte
from flyte import Image
from flyte._internal.imagebuild.image_builder import ImageBuildEngine

# Base image is now defined at the environment level
env = flyte.TaskEnvironment(
    name="benchmark",
    image=(Image.from_debian_base(name="benchmark-base").with_pip_packages("torch", "numpy", "pandas")),
)


@env.task
async def benchmark_layer_optimization() -> Dict[str, float]:
    print("Starting Docker layer optimization benchmark")

    # No optimization
    image_no_opt = Image.from_debian_base(name="benchmark-no-opt").with_pip_packages(
        "torch", "numpy", "pandas", "requests"
    )

    start = time.time()
    await ImageBuildEngine.build(image_no_opt, force=True, optimize_layers=False)
    no_opt_time = time.time() - start
    print(f"No optimization build: {no_opt_time:.1f}s")

    # Phase 3: With optimization
    image_opt = Image.from_debian_base(name="benchmark-opt").with_pip_packages("torch", "numpy", "pandas", "httpx")

    start = time.time()
    await ImageBuildEngine.build(image_opt, force=True, optimize_layers=True)
    opt_time = time.time() - start
    print(f"Optimized build: {opt_time:.1f}s")

    speedup = no_opt_time / opt_time if opt_time > 0 else 1.0

    print(f"no-opt={no_opt_time:.1f}s | opt={opt_time:.1f}s | speedup={speedup:.1f}x")

    return {
        "no_opt_time": no_opt_time,
        "opt_time": opt_time,
        "speedup": speedup,
    }


if __name__ == "__main__":
    flyte.init_from_config(log_level=logging.DEBUG)
    run = flyte.with_runcontext(mode="remote", log_level=logging.DEBUG).run(benchmark_layer_optimization)
    print(run.name)
    print(run.url)
