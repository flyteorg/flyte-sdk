"""
Flyte extras package.
This package provides various utilities that make it possible to build highly customized workflows.

1. ContainerTask: Execute arbitrary pre-containerized applications, without needed the `flyte-sdk` to be installed.
                  This extra uses `flyte copilot` system to inject inputs and slurp outputs from the container run.

2. Time utilities: Usage of Time.now, time.sleep or asyncio.sleep bring non-determinism to a program. This module
                   provides a few utilities that make it possible to bring determinism to workflows that need to access
                   time related functions. This determinism persists across crashes and restarts making the process
                   durable.
"""

from flyte.durable._time import durable_sleep, durable_time

from ._container import ContainerTask

__all__ = [
    "ContainerTask",
    "durable_sleep",
    "durable_time",
]
