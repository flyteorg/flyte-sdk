"""Task plugin for the batch job connector."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from flyte.connectors import AsyncConnectorExecutorMixin
from flyte.extend import TaskTemplate
from flyte.models import NativeInterface, SerializationContext


@dataclass
class BatchJobConfig:
    """Configuration for a batch job task.

    Attributes:
        timeout_seconds: Maximum time in seconds to wait for the job to complete.
    """

    timeout_seconds: int = 300


class BatchJobTask(AsyncConnectorExecutorMixin, TaskTemplate):
    _TASK_TYPE = "batch_job"

    def __init__(
        self,
        name: str,
        plugin_config: BatchJobConfig,
        inputs: Optional[Dict[str, Type]] = None,
        outputs: Optional[Dict[str, Type]] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            interface=NativeInterface(
                {k: (v, None) for k, v in inputs.items()} if inputs else {},
                outputs or {},
            ),
            task_type=self._TASK_TYPE,
            image=None,
            **kwargs,
        )
        self.plugin_config = plugin_config

    def custom_config(self, sctx: SerializationContext) -> Optional[Dict[str, Any]]:
        return {"timeout_seconds": self.plugin_config.timeout_seconds}
