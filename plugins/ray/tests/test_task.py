from pathlib import Path

import flyte
import pytest
from flyte import PodTemplate, Resources
from flyte.models import SerializationContext
from flyteidl2.core import tasks_pb2
from flyteidl2.plugins.ray_pb2 import RayJob
from google.protobuf.json_format import MessageToDict, ParseDict
from kubernetes.client import V1Container, V1PodSpec, V1ResourceRequirements

from flyteplugins.ray.task import HeadNodeConfig, RayFunctionTask, RayJobConfig, WorkerNodeConfig


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


def _primary_container(k8s_pod) -> dict:
    """Return the first container of a serialized k8s_pod as a plain dict."""
    pod_spec = MessageToDict(k8s_pod.pod_spec)
    return pod_spec["containers"][0]


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


def _wut_pod_template(container_name: str) -> PodTemplate:
    """A pod template that sets a custom entrypoint and cpu/memory resources."""
    return PodTemplate(
        pod_spec=V1PodSpec(
            containers=[
                V1Container(
                    name=container_name,
                    args=["wut update-aws-credentials-file default"],
                    resources=V1ResourceRequirements(
                        requests={"cpu": "15000m", "memory": "45Gi"},
                        limits={"cpu": "15000m", "memory": "45Gi"},
                    ),
                )
            ]
        )
    )


def test_worker_pod_template_merged_with_requests(sctx):
    """requests/limits must merge into the worker pod_template instead of replacing it.

    Regression test: previously, setting requests/limits caused the SDK to build a
    fresh pod spec from resources and drop the user's PodTemplate (losing custom
    args/command/env and cpu/memory).
    """
    ray_config = RayJobConfig(
        worker_node_config=[
            WorkerNodeConfig(
                group_name="workers",
                replicas=2,
                requests=Resources(gpu="A10G:1"),
                pod_template=_wut_pod_template("ray-worker"),
            ),
        ],
    )
    ray_task = _build_ray_task(ray_config)

    ray_job = _to_ray_job(ray_task.custom_config(sctx))

    worker_spec = ray_job.ray_cluster.worker_group_spec[0]
    container = _primary_container(worker_spec.k8s_pod)

    # Custom entrypoint from the pod_template is preserved.
    assert container["args"] == ["wut update-aws-credentials-file default"]
    # cpu/memory from the pod_template AND the gpu from requests are both present.
    assert container["resources"]["requests"]["cpu"] == "15000m"
    assert container["resources"]["requests"]["memory"] == "45Gi"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    # extended_resources still reflects the requested accelerator type.
    assert worker_spec.extended_resources.gpu_accelerator.device == "nvidia-a10g"


def test_head_pod_template_merged_with_requests(sctx):
    """requests/limits must merge into the head pod_template instead of replacing it."""
    ray_config = RayJobConfig(
        head_node_config=HeadNodeConfig(
            requests=Resources(gpu="T4:1"),
            pod_template=_wut_pod_template("ray-head"),
        ),
        worker_node_config=[WorkerNodeConfig(group_name="grp", replicas=1)],
    )
    ray_task = _build_ray_task(ray_config)

    ray_job = _to_ray_job(ray_task.custom_config(sctx))

    head_spec = ray_job.ray_cluster.head_group_spec
    container = _primary_container(head_spec.k8s_pod)

    assert container["args"] == ["wut update-aws-credentials-file default"]
    assert container["resources"]["requests"]["cpu"] == "15000m"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert head_spec.extended_resources.gpu_accelerator.device == "nvidia-tesla-t4"


def test_requests_take_precedence_over_pod_template_resources(sctx):
    """Resource keys from requests/limits override the same keys on the pod_template."""
    ray_config = RayJobConfig(
        worker_node_config=[
            WorkerNodeConfig(
                group_name="workers",
                replicas=1,
                requests=Resources(cpu="8000m", memory="32Gi"),
                pod_template=_wut_pod_template("ray-worker"),
            ),
        ],
    )
    ray_task = _build_ray_task(ray_config)

    ray_job = _to_ray_job(ray_task.custom_config(sctx))

    container = _primary_container(ray_job.ray_cluster.worker_group_spec[0].k8s_pod)
    # requests=Resources(cpu="8000m", memory="32Gi") overrides the template's 15000m/45Gi.
    assert container["resources"]["requests"]["cpu"] == "8000m"
    assert container["resources"]["requests"]["memory"] == "32Gi"
    # The custom entrypoint is still preserved.
    assert container["args"] == ["wut update-aws-credentials-file default"]


def test_pod_template_without_resources_is_unchanged(sctx):
    """A pod_template with no requests/limits passes through untouched (existing behavior)."""
    ray_config = RayJobConfig(
        worker_node_config=[
            WorkerNodeConfig(
                group_name="workers",
                replicas=1,
                pod_template=_wut_pod_template("ray-worker"),
            ),
        ],
    )
    ray_task = _build_ray_task(ray_config)

    ray_job = _to_ray_job(ray_task.custom_config(sctx))

    container = _primary_container(ray_job.ray_cluster.worker_group_spec[0].k8s_pod)
    assert container["args"] == ["wut update-aws-credentials-file default"]
    assert container["resources"]["requests"]["cpu"] == "15000m"


def test_custom_config_records_reuse_policy(sctx):
    task = RayFunctionTask(
        name="t",
        interface=None,
        func=lambda: None,
        plugin_config=RayJobConfig(worker_node_config=[]),
        reusable=flyte.ReusePolicy(replicas=1, idle_ttl=600),
    )
    custom = task.custom_config(sctx)
    assert custom["reusePolicy"] == {
        "parallelism": 1,
        "min_replica_count": 1,
        "replica_count": 1,
        "ttl_seconds": 600,
        "scaledown_ttl_seconds": 30,
    }
    # The rest of the spec still parses as a RayJob (the extra field is ignored by the proto).
    assert "rayCluster" in custom


def test_custom_config_reuse_replicas_map_to_worker_scaling(sctx):
    task = RayFunctionTask(
        name="t",
        interface=None,
        func=lambda: None,
        plugin_config=RayJobConfig(worker_node_config=[]),
        reusable=flyte.ReusePolicy(replicas=(1, 3), scaledown_ttl=60),
    )
    custom = task.custom_config(sctx)
    # replicas maps to the shared cluster's worker scaling — multiple replicas are allowed.
    assert custom["reusePolicy"]["min_replica_count"] == 1
    assert custom["reusePolicy"]["replica_count"] == 3
    assert custom["reusePolicy"]["scaledown_ttl_seconds"] == 60
