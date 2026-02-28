"""
Flyte extras package.
This package provides various utilities that make it possible to build highly customized workflows.

1. ContainerTask: Execute arbitrary pre-containerized applications, without needing the `flyte-sdk`
                  to be installed. This extra uses `flyte copilot` system to inject inputs and slurp
                  outputs from the container run.

2. Time utilities: Usage of Time.now, time.sleep or asyncio.sleep bring non-determinism to a program.
                   This module provides a few utilities that make it possible to bring determinism to
                   workflows that need to access time related functions. This determinism persists
                   across crashes and restarts making the process durable.

3. GPUSaturator: Maximize GPU utilization by batching work from many concurrent producers through a
                 single async inference function.  Useful for large-scale batch inference with
                 reusable containers.
"""

from flyte.durable._time import durable_sleep, durable_time

from ._container import ContainerTask
from ._gpu_saturator import BatchStats, GPUSaturator, TokenEstimator

__all__ = [
    "BatchStats",
    "ContainerTask",
    "GPUSaturator",
    "TokenEstimator",
    "durable_sleep",
    "durable_time",
]
