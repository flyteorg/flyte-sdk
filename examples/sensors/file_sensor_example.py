"""
File Sensor Example

This example demonstrates how to use the FileSensor to detect files
appearing in your local or remote filesystem.

The FileSensor polls for the existence of a file at a given path
and completes when the file is found.
"""

from datetime import timedelta

import flyte
from flyte.sensors import FileSensor

# Create a file sensor that will wait for a file to appear
# You can optionally specify a timeout
file_sensor = FileSensor(
    name="wait_for_data",
    timeout=timedelta(minutes=30),  # Wait up to 30 minutes
)

# Create a TaskEnvironment for the sensor using from_task()
# This is required for standalone connector-based tasks
sensor_env = flyte.TaskEnvironment.from_task("sensor_env", file_sensor)

# Create the main task environment that depends on the sensor environment
env = flyte.TaskEnvironment(name="sensor_example", depends_on=[sensor_env])


@env.task
async def process_data(path: str) -> str:
    """Process the data file once it's available."""
    print(f"Processing data from: {path}")
    # In a real scenario, you would read and process the file here
    return f"Successfully processed data from {path}"


@env.task
async def main(data_path: str) -> str:
    """
    Main task that waits for a file and then processes it.

    This demonstrates the file sensor pattern where we:
    1. Wait for a file to appear using the sensor
    2. Process the file once it's available
    """
    print(f"Waiting for file: {data_path}")

    # The sensor will block until the file exists
    await file_sensor(path=data_path)

    print("File found! Starting processing...")

    # Now process the data
    result = await process_data(data_path)
    return result


if __name__ == "__main__":
    import sys
    import tempfile
    from pathlib import Path

    # For local testing, we'll create a temp file
    # In production, you'd wait for a file from S3/GCS/etc.

    flyte.init_from_config()

    # Get the path from command line or use a default
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        print(f"Using provided path: {test_path}")
    else:
        # Create a temp directory and file for testing
        temp_dir = tempfile.mkdtemp()
        test_path = str(Path(temp_dir) / "test_data.csv")

        # Create the file immediately for testing
        # In a real scenario, the file would be created by another process
        print(f"Creating test file: {test_path}")
        with open(test_path, "w") as f:
            f.write("col1,col2,col3\n1,2,3\n4,5,6\n")

    # test_path = "s3://daniel-sola-testing/test.txt"
    # Run the task
    print("\n--- Running file sensor example ---\n")
    run = flyte.with_runcontext(mode="local").run(main, data_path=test_path)
    print(f"\nResult: {run}")