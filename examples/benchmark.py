"""
Benchmark:  Layer optimization cache test.
"""

import logging
import time

import flyte
from flyte import Image

env = flyte.TaskEnvironment(name="benchmark")


@env.task
async def main():
    image1 = (
        Image.from_debian_base(name="benchmark")
        .with_pip_packages("tensorflow", "torch", "numpy", "pandas", optimize_layers=False)
        .with_env_vars({"build": "1"})
    )

    # WITHOUT OPTIMIZATION
    image1 = (
        Image.from_debian_base(name="benchmark")
        .with_pip_packages("tensorflow", "torch", "numpy", "pandas", optimize_layers=False)
        .with_env_vars({"build": "1"})
    )

    start = time.time()
    await flyte.build.aio(image1)

    image2 = (
        Image.from_debian_base(name="benchmark")
        .with_pip_packages("tensorflow", "torch", "numpy", "pandas", "matplotlib", "pytest", optimize_layers=False)
        .with_env_vars({"build": "2"})
    )

    start = time.time()
    await flyte.build.aio(image2)
    time2 = time.time() - start

    # WITH OPTIMIZATION
    image3 = (
        Image.from_debian_base(name="benchmark")
        .with_pip_packages("tensorflow", "torch", "numpy", "pandas", optimize_layers=True)
        .with_env_vars({"build": "3"})
    )

    start = time.time()
    await flyte.build.aio(image3)

    image4 = (
        Image.from_debian_base(name="benchmark")
        .with_pip_packages("tensorflow", "torch", "numpy", "pandas", "matplotlib", "pytest", optimize_layers=True)
        .with_env_vars({"build": "4"})
    )

    start = time.time()
    await flyte.build.aio(image4)
    time4 = time.time() - start
    
    print(f"\nNo opt:  {time2:.1f}s | With opt: {time4:.1f}s | Speedup: {time2 / time4:.1f}x\n")

    assert time4 < time2
    print(image1)
    print(image2)
    print(image3)
    print(image4)


if __name__ == "__main__":
    flyte.init_from_config(log_level=logging.DEBUG)
    run = flyte.with_runcontext(mode="remote", log_level=logging.DEBUG).run(main)
    print(run.name)
    print(run.url)
