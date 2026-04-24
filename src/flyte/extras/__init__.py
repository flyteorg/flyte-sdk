"""
Flyte extras package.
This package provides various utilities that make it possible to build highly customized workflows.

1. ContainerTask: Execute arbitrary pre-containerized applications, without needing the `flyte-sdk`
                  to be installed. This extra uses `flyte copilot` system to inject inputs and slurp
                  outputs from the container run.

2. DynamicBatcher / TokenBatcher: Maximize resource utilization by batching work from many concurrent
                   producers through a single async processing function.  DynamicBatcher is the
                   general-purpose base; TokenBatcher is a convenience subclass for token-budgeted
                   LLM inference with reusable containers.

3. Sleep: Route a task to the backend `core-sleep` plugin, which executes in leaseworker with no
                   task pod.
"""

from ._container import ContainerTask
from ._dynamic_batcher import (
    BatchStats,
    CostEstimator,
    DynamicBatcher,
    Prompt,
    TokenBatcher,
    TokenEstimator,
)
from ._sleep import Sleep, SleepTask

__all__ = [
    "BatchStats",
    "ContainerTask",
    "CostEstimator",
    "DynamicBatcher",
    "Prompt",
    "Sleep",
    "SleepTask",
    "TokenBatcher",
    "TokenEstimator",
]
