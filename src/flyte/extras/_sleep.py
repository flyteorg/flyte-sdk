from __future__ import annotations

from dataclasses import dataclass

from flyte.extend import AsyncFunctionTaskTemplate, TaskPluginRegistry


@dataclass
class Sleep:
    """
    Route a task to the backend `core-sleep` plugin.

    The sleep duration is provided as a normal task input, not plugin config.
    """


@dataclass(kw_only=True)
class SleepTask(AsyncFunctionTaskTemplate):
    plugin_config: Sleep
    task_type: str = "core-sleep"


TaskPluginRegistry.register(Sleep, SleepTask)
