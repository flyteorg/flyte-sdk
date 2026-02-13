"""
Sensors for Flyte v2.

Sensors are tasks that poll for conditions to be met before completing.
They are useful for waiting on external events like files appearing
in storage systems.

Example usage:

```python
import flyte
from flyte.sensors import FileSensor

# Create a file sensor
file_sensor = FileSensor(name="wait_for_data")

# Register the sensor with a TaskEnvironment
sensor_env = flyte.TaskEnvironment.from_task("sensor_env", file_sensor)

# Create main environment that depends on the sensor environment
env = flyte.TaskEnvironment(name="my_env", depends_on=[sensor_env])

@env.task
async def main():
    # Wait for the file to appear
    await file_sensor(path="s3://my-bucket/data.csv")

    # Process the data now that it exists
    print("File found! Processing...")
```
"""

from ._file_sensor import FileSensor, FileSensorConfig
from ._sensor_connector import SensorConnector, SensorMetadata

__all__ = [
    "FileSensor",
    "FileSensorConfig",
    "SensorConnector",
    "SensorMetadata",
]