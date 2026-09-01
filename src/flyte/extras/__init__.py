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

4. IdempotentRun: Launch a run once per event key, for webhook receivers, pollers, and
                  anything else outside Flyte that reacts to events which may arrive twice.

5. Shell: Wrap a CLI tool packaged in a container image. Designed as the foundation for
                bio module libraries (bedtools, samtools, bcftools, GATK, etc.) and any other case
                where a user wants to call a pre-built binary in a published container with
                typed inputs and outputs.
"""

from . import shell
from ._container import ContainerTask
from ._dynamic_batcher import (
    BatchStats,
    CostEstimator,
    DynamicBatcher,
    Prompt,
    TokenBatcher,
    TokenEstimator,
)
from ._idempotent_run import DUPE_LABEL_KEY, DuplicateRun, blocking_run, idempotent_run
from ._serialize import serialize, serialize_env
from ._sleep import Sleep, SleepTask

__all__ = [
    "DUPE_LABEL_KEY",
    "BatchStats",
    "ContainerTask",
    "CostEstimator",
    "DuplicateRun",
    "DynamicBatcher",
    "Prompt",
    "Sleep",
    "SleepTask",
    "TokenBatcher",
    "TokenEstimator",
    "blocking_run",
    "idempotent_run",
    "serialize",
    "serialize_env",
    "shell",
]
