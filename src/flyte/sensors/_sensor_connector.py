"""
Sensor connector for polling-based sensors in Flyte v2.

This connector handles the polling lifecycle for sensors, checking conditions
periodically until they are met.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import flyte.storage as storage
from flyte.connectors import AsyncConnector, ConnectorRegistry, Resource, ResourceMeta
from flyteidl2.core.execution_pb2 import TaskExecution
from flyteidl2.core.tasks_pb2 import TaskTemplate
from google.protobuf import json_format


@dataclass
class SensorMetadata(ResourceMeta):
    """
    Metadata for tracking sensor state between polling calls.
    """

    sensor_type: str
    path: Optional[str] = None


class SensorConnector(AsyncConnector):
    """
    Connector for sensors that poll for conditions to be met.

    This connector handles file sensors and can be extended to support
    other sensor types in the future.
    """

    name = "Sensor Connector"
    task_type_name = "sensor"
    metadata_type = SensorMetadata

    async def create(
        self,
        task_template: TaskTemplate,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SensorMetadata:
        """
        Initialize the sensor with the path to watch.
        """
        custom = json_format.MessageToDict(task_template.custom) if task_template.custom else {}
        sensor_type = custom.get("sensor_type", "file")

        # Get the path from inputs
        path = inputs.get("path") if inputs else None

        return SensorMetadata(sensor_type=sensor_type, path=path)

    async def get(self, resource_meta: SensorMetadata, **kwargs) -> Resource:
        """
        Check if the sensor condition is met.

        For file sensors, this checks if the file exists at the given path.
        Returns SUCCEEDED if condition is met, RUNNING otherwise.
        """
        if resource_meta.sensor_type == "file":
            if resource_meta.path is None:
                return Resource(
                    phase=TaskExecution.FAILED,
                    message="No path provided for file sensor",
                )

            try:
                exists = await storage.exists(resource_meta.path)
                if exists:
                    return Resource(
                        phase=TaskExecution.SUCCEEDED,
                        message=f"File found: {resource_meta.path}",
                    )
                return Resource(
                    phase=TaskExecution.RUNNING,
                    message=f"Waiting for file: {resource_meta.path}",
                )
            except Exception as e:
                return Resource(
                    phase=TaskExecution.RUNNING,
                    message=f"Error checking file existence: {e}",
                )

        return Resource(
            phase=TaskExecution.FAILED,
            message=f"Unknown sensor type: {resource_meta.sensor_type}",
        )

    async def delete(self, resource_meta: SensorMetadata, **kwargs):
        """
        No-op for sensors - nothing to cancel.
        """
        pass


# Register the connector
ConnectorRegistry.register(SensorConnector())