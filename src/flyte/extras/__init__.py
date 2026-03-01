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

3. DynamicBatcher / TokenBatcher: Maximize resource utilization by batching work from many concurrent
                   producers through a single async processing function.  DynamicBatcher is the
                   general-purpose base; TokenBatcher is a convenience subclass for token-budgeted
                   LLM inference with reusable containers.
"""

from flyte.durable._time import durable_sleep, durable_time

from ._container import ContainerTask
from ._dynamic_batcher import (
    BatchStats,
    CostEstimator,
    DynamicBatcher,
    Prompt,
    TokenBatcher,
    TokenEstimator,
)

__all__ = [
    "BatchStats",
    "ContainerTask",
    "CostEstimator",
    "DynamicBatcher",
    "Prompt",
    "TokenBatcher",
    "TokenEstimator",
    "durable_sleep",
    "durable_time",
]
