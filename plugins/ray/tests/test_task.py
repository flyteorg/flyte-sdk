from pathlib import Path

import pytest
from flyteidl2.core import tasks_pb2
from flyteidl2.plugins.ray_pb2 import RayJob
from google.protobuf.json_format import ParseDict

import flyte
from flyte import Resources
from flyte.models import SerializationContext
from flyteplugins.ray.task import HeadNodeConfig, RayJobConfig, WorkerNodeConfig


@pytest.fixture
def sctx() -> SerializationContext:
    return SerializationContext(
        code_bundle=None,
        version="abc123",
        input_path="s3://bucket/test/run/inputs.pb",
        output_path="s3://bucket/outputs/0/jfkljfa/0",
        root_dir=Path.cwd(),
    )


def _build_ray_task(ray_config: RayJobConfig):
    env = flyte.TaskEnvironment(name="ray_env", plugin_config=ray_config)

    @env.task
    async def my_ray_task() -> int:
        return 1

    return my_ray_task


def _to_ray_job(custom_config: dict) -> RayJob:
    return ParseDict(custom_config, RayJob())


def test_head_node_extended_resources(sctx):
    """extended_resources is populated from head_node_config.requests GPU accelerator."""
    ray_config = RayJobConfig(
        head_node_config=HeadNodeConfig(requests=Resources(cpu=1, gpu="T4:1")),
        worker_node_config=[WorkerNodeConfig(group_name="grp", replicas=1)],
    )
    ray_task = _build_ray_task(ray_config)

    ray_job = _to_ray_job(ray_task.custom_config(sctx))

    head_spec = ray_job.ray_cluster.head_group_spec
    accelerator = head_spec.extended_resources.gpu_accelerator
    assert accelerator.device == "nvidia-tesla-t4"
    assert accelerator.device_class == tasks_pb2.GPUAccelerator.NVIDIA_GPU


def test_worker_group_extended_resources(sctx):
    """extended_resources is populated from each worker_node_config.requests GPU accelerator."""
    ray_config = RayJobConfig(
        worker_node_config=[
            WorkerNodeConfig(group_name="grp", replicas=2, requests=Resources(cpu=1, gpu="A100:2")),
        ],
    )
    ray_task = _build_ray_task(ray_config)

    ray_job = _to_ray_job(ray_task.custom_config(sctx))

    worker_specs = ray_job.ray_cluster.worker_group_spec
    assert len(worker_specs) == 1
    accelerator = worker_specs[0].extended_resources.gpu_accelerator
    assert accelerator.device == "nvidia-tesla-a100"
    assert accelerator.device_class == tasks_pb2.GPUAccelerator.NVIDIA_GPU


def test_extended_resources_absent_without_gpu(sctx):
    """No extended_resources is set when requests do not include an accelerator."""
    ray_config = RayJobConfig(
        head_node_config=HeadNodeConfig(requests=Resources(cpu=1, memory="1Gi")),
        worker_node_config=[
            WorkerNodeConfig(group_name="grp", replicas=1, requests=Resources(cpu=1, memory="1Gi")),
        ],
    )
    ray_task = _build_ray_task(ray_config)

    ray_job = _to_ray_job(ray_task.custom_config(sctx))

    assert not ray_job.ray_cluster.head_group_spec.HasField("extended_resources")
    assert not ray_job.ray_cluster.worker_group_spec[0].HasField("extended_resources")
