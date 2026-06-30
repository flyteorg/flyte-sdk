from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flyte.extend import AsyncFunctionTaskTemplate, TaskPluginRegistry
from flyte.models import SerializationContext


@dataclass
class Echo:
    """
    Route a task to the backend `echo` plugin.

    The backend plugin is responsible for execution behavior. The SDK wrapper only
    marks the serialized task as type `echo`.
    """


@dataclass(kw_only=True)
class EchoTask(AsyncFunctionTaskTemplate):
    plugin_config: Echo
    task_type: str = "echo"

    def custom_config(self, sctx: SerializationContext) -> dict[str, Any]:
        return {}


TaskPluginRegistry.register(Echo, EchoTask)
