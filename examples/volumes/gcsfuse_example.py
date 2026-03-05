# /// script
# requires-python = "==3.12"
# dependencies = [
#    "kubernetes",
#    "flyte>2.0.0b39",
# ]
# ///
"""
GCSFuse Example - 2-Task Workflow

Demonstrates using gcsfuse to share data between tasks:
1. Producer task writes data to gcsfuse-mounted GCS bucket
2. Consumer task reads data from the same bucket
3. Both tasks share the same TaskEnvironment with pod_template mounting gcsfuse

Prerequisites:
- GCS bucket must exist (configure bucket name below)
- Kubernetes cluster must have GCSFuse CSI driver installed
- Appropriate GCP service account permissions

Usage:
    flyte run gcsfuse_example.py gcsfuse_workflow --data="Hello GCS!"
"""

import os

from kubernetes.client import (
    V1Container,
    V1CSIVolumeSource,
    V1PodSpec,
    V1Volume,
    V1VolumeMount,
)

import flyte
from flyte.io import File

# Configure your GCS bucket name here
GCS_BUCKET = os.environ.get("MY_BUCKET", "opta-gcp-dogfood-gcp")
MOUNT_PATH = "/mnt/gcs"

# Create pod template with gcsfuse volume mount
# Note: This uses CSI volume source for gcsfuse integration
pod_template = flyte.PodTemplate(
    primary_container_name="primary",
    annotations={
        # Annotations that help GKE mount the right sidecar.
        "gke-gcsfuse/volumes": "true",
        "gke-gcsfuse/cpu-limit": "4",
        "gke-gcsfuse/memory-limit": "2Gi",
        "gke-gcsfuse/ephemeral-storage-limit": "100Gi",
        "gke-gcsfuse/cpu-request": "500m",
        "gke-gcsfuse/memory-request": "1Gi",
        "gke-gcsfuse/ephemeral-storage-request": "10Gi",
    },
    pod_spec=V1PodSpec(
        containers=[
            V1Container(
                name="primary",
                volume_mounts=[
                    V1VolumeMount(
                        name="gcs-fuse",
                        mount_path=MOUNT_PATH,
                        read_only=False,
                    )
                ],
            )
        ],
        volumes=[
            V1Volume(
                name="gcs-fuse",
                csi=V1CSIVolumeSource(
                    driver="gcsfuse.csi.storage.gke.io",
                    volume_attributes={
                        "bucketName": GCS_BUCKET,
                        # Match the uid and gid as set by the union remote image build system.
                        "mountOptions": "implicit-dirs,uid=65532,gid=65532",
                    },
                ),
            )
        ],
    ),
)

# Create shared TaskEnvironment with gcsfuse pod template
env = flyte.TaskEnvironment(
    name="gcsfuse-wf",
    pod_template=pod_template,
    image=flyte.Image.from_uv_script(__file__, name="jeevs-env", pre=True),
)


@env.task
async def produce_data(content: str) -> File:
    """
    Task 1: Producer - Writes data to gcsfuse-mounted GCS bucket.

    Args:
        content: String content to write to the file

    Returns:
        File reference to the created file in GCS
    """
    # Write directly to the gcsfuse mount
    # This automatically syncs to GCS via gcsfuse
    file_path = f"{MOUNT_PATH}/data.txt"

    print(f"Writing data to gcsfuse mount: {file_path}")
    with open(file_path, "w") as f:  # noqa: ASYNC230
        f.write(content)

    print(f"Data written successfully: '{content}'")

    # Create File reference pointing to the GCS location
    # The file already exists in GCS via the gcsfuse mount
    gcs_uri = f"gs://{GCS_BUCKET}/data.txt"
    file_ref = File.from_existing_remote(gcs_uri)

    print(f"Created file reference: {gcs_uri}")
    return file_ref


@env.task
async def consume_data(f: File) -> str:
    """
    Task 2: Consumer - Reads data from the File reference.

    The file is accessible via the gcsfuse mount since both tasks
    share the same TaskEnvironment with the gcsfuse pod template.

    Args:
        f: File reference from the producer task

    Returns:
        String content read from the file
    """
    # Read directly from gcsfuse mount
    # Since we know the file is at a fixed location, we can read it directly
    file_path = f"{MOUNT_PATH}/data.txt"

    print(f"Reading data from gcsfuse mount: {file_path}")
    with open(file_path, "r") as fh:  # noqa: ASYNC230
        content = fh.read()

    print(f"Data read successfully: '{content}'")

    # Alternative: Could also download via the File reference
    # local_path = await f.download()
    # with open(local_path, "r") as fh:
    #     content = fh.read()

    return content


@env.task
async def gcsfuse_workflow(data: str = "Hello from GCSFuse!") -> str:
    """
    Main workflow demonstrating 2-task data passing via gcsfuse.

    This workflow:
    1. Calls produce_data to write data to GCS via gcsfuse
    2. Calls consume_data to read the data back via gcsfuse
    3. Returns the read content

    Args:
        data: Content to write and read back

    Returns:
        The content read from the second task (should match input)
    """
    print(f"Starting gcsfuse workflow with data: '{data}'")

    # Task 1: Produce data
    print("\n=== Task 1: Producing data ===")
    file_ref = await produce_data(data)

    # Task 2: Consume data
    print("\n=== Task 2: Consuming data ===")
    result = await consume_data(file_ref)

    print("\n=== Workflow complete ===")
    print(f"Input: '{data}'")
    print(f"Output: '{result}'")
    print(f"Match: {data == result}")

    return result


if __name__ == "__main__":
    flyte.init_from_config()
    result = flyte.run(gcsfuse_workflow, data="Hello from GCSFuse!")
    print(result.url)
