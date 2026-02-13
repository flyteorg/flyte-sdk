"""
FileSensor task for detecting files in local or remote filesystems.

This sensor polls for the existence of a file at a given path and completes
when the file is found.
"""

from __future__ import annotations

import inspect
import weakref
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from flyte.connectors import AsyncConnectorExecutorMixin
from flyte.extend import TaskTemplate
from flyte.models import NativeInterface, SerializationContext

if TYPE_CHECKING:
    from flyte import TaskEnvironment


@dataclass
class FileSensorConfig:
    """
    Configuration for the FileSensor.

    :param timeout: Maximum time to wait for the file to appear.
                   Can be specified as seconds (int) or timedelta.
    """

    timeout: Optional[Union[int, timedelta]] = None


class FileSensor(AsyncConnectorExecutorMixin, TaskTemplate):
    """
    A sensor that detects files appearing in local or remote filesystems.

    The FileSensor polls for the existence of a file at the given path
    and completes when the file is found. It supports local files as well
    as remote storage systems like S3, GCS, and Azure Blob Storage.

    Example usage:

    ```python
    import flyte
    from flyte.sensors import FileSensor

    # Create a file sensor and associate it with an environment
    file_sensor = FileSensor(name="wait_for_data")
    sensor_env = flyte.TaskEnvironment.from_task("sensor_env", file_sensor)

    env = flyte.TaskEnvironment(name="my_env", depends_on=[sensor_env])

    @env.task
    async def main():
        # Wait for the file to appear
        await file_sensor(path="s3://my-bucket/data.csv")

        # Process the data now that it exists
        print("File found! Processing...")
    ```

    You can also specify a timeout:

    ```python
    from datetime import timedelta

    file_sensor = FileSensor(
        name="wait_for_data",
        timeout=timedelta(hours=1),  # Wait up to 1 hour
    )
    ```

    :param name: The name of the sensor task.
    :param timeout: Optional maximum time to wait for the file.
                   Can be seconds (int) or timedelta.
    :param env_name: Optional name for auto-created TaskEnvironment.
                    If not provided, uses "{name}_env".
    :param kwargs: Additional arguments passed to the TaskTemplate.
    """

    _TASK_TYPE = "sensor"

    def __init__(
        self,
        name: str,
        timeout: Optional[Union[int, timedelta]] = None,
        env_name: Optional[str] = None,
        **kwargs,
    ):
        # Convert timeout to timedelta if needed
        timeout_td = None
        if timeout is not None:
            if isinstance(timeout, int):
                timeout_td = timedelta(seconds=timeout)
            else:
                timeout_td = timeout

        super().__init__(
            name=name,
            interface=NativeInterface(
                inputs={"path": (str, inspect.Parameter.empty)},
                outputs={},
            ),
            task_type=self._TASK_TYPE,
            timeout=timeout_td,
            **kwargs,
        )
        self._sensor_config = FileSensorConfig(timeout=timeout)
        self._env_name = env_name

    def register_with_environment(self, env: TaskEnvironment) -> None:
        """
        Register this sensor with a TaskEnvironment.

        This is called automatically when using TaskEnvironment.from_task().
        """
        self.parent_env = weakref.ref(env)
        self.parent_env_name = env.name

    def custom_config(self, sctx: SerializationContext) -> Optional[Dict[str, Any]]:
        """
        Return custom configuration for the sensor connector.
        """
        config: Dict[str, Any] = {
            "sensor_type": "file",
        }
        if self._sensor_config.timeout is not None:
            if isinstance(self._sensor_config.timeout, timedelta):
                config["timeout_seconds"] = int(self._sensor_config.timeout.total_seconds())
            else:
                config["timeout_seconds"] = self._sensor_config.timeout
        return config