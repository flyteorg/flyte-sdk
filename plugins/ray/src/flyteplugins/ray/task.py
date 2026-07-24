import base64
import copy
import json
import os
import typing
from dataclasses import dataclass
from typing import Any, Dict, Optional

import flyte
import flyte.errors
import yaml
from flyte import PodTemplate, Resources
from flyte.extend import (
    AsyncFunctionTaskTemplate,
    TaskPluginRegistry,
    get_proto_extended_resources,
    pod_spec_from_resources,
)
from flyte.models import SerializationContext
from flyteidl2.plugins.ray_pb2 import HeadGroupSpec, RayCluster, RayJob, WorkerGroupSpec
from google.protobuf.json_format import MessageToDict

import ray

if typing.TYPE_CHECKING:
    pass


_RAY_HEAD_CONTAINER_NAME = "ray-head"
_RAY_WORKER_CONTAINER_NAME = "ray-worker"


def _build_node_pod_template(
    primary_container_name: str,
    pod_template: Optional[PodTemplate],
    requests: Optional[Resources],
    limits: Optional[Resources],
) -> Optional[PodTemplate]:
    """
    Build the K8s pod template for a Ray head/worker group.

    When ``requests``/``limits`` are set they are *merged* into the primary container of the
    user-supplied ``pod_template`` rather than replacing it, so custom fields such as
    ``args``/``command``/``env``/volumes set on the template are preserved. Resource keys derived
    from ``requests``/``limits`` take precedence over any already present on the primary container.

    If no ``pod_template`` is provided, a pod spec is built from the resources alone. If neither
    ``requests`` nor ``limits`` is set, the ``pod_template`` is returned unchanged.
    """
    if not requests and not limits:
        return pod_template

    from kubernetes.client import V1Container, V1ResourceRequirements

    # Resource requirements derived from the structured Resources (handles the nvidia.com/gpu key,
    # singular-resource validation and request/limit mirroring) for the primary container.
    resource_pod_spec = pod_spec_from_resources(
        primary_container_name=primary_container_name,
        requests=requests,
        limits=limits,
    )
    resource_requirements = resource_pod_spec.containers[0].resources

    # No user-supplied spec to merge into: fall back to the resource-only pod spec, preserving any
    # labels/annotations/primary_container_name carried by the template.
    if pod_template is None or pod_template.pod_spec is None:
        return PodTemplate(
            pod_spec=resource_pod_spec,
            primary_container_name=pod_template.primary_container_name if pod_template else primary_container_name,
            labels=pod_template.labels if pod_template else None,
            annotations=pod_template.annotations if pod_template else None,
        )

    merged = copy.deepcopy(pod_template)
    containers = list(merged.pod_spec.containers or [])

    # Locate the container the resources belong to: prefer the Ray container name, otherwise the
    # sole container, otherwise append a new one.
    primary = next((c for c in containers if c.name == primary_container_name), None)
    if primary is None and len(containers) == 1:
        primary = containers[0]
    if primary is None:
        primary = V1Container(name=primary_container_name)
        containers.append(primary)
        merged.pod_spec.containers = containers

    existing = primary.resources or V1ResourceRequirements()
    primary.resources = V1ResourceRequirements(
        requests={**(existing.requests or {}), **(resource_requirements.requests or {})} or None,
        limits={**(existing.limits or {}), **(resource_requirements.limits or {})} or None,
    )
    return merged


@dataclass
class HeadNodeConfig:
    ray_start_params: typing.Optional[typing.Dict[str, str]] = None
    pod_template: typing.Optional[PodTemplate] = None
    requests: Optional[Resources] = None
    limits: Optional[Resources] = None


@dataclass
class WorkerNodeConfig:
    group_name: str
    replicas: int
    min_replicas: typing.Optional[int] = None
    max_replicas: typing.Optional[int] = None
    ray_start_params: typing.Optional[typing.Dict[str, str]] = None
    pod_template: typing.Optional[PodTemplate] = None
    requests: Optional[Resources] = None
    limits: Optional[Resources] = None


@dataclass
class RayJobConfig:
    worker_node_config: typing.List[WorkerNodeConfig]
    head_node_config: typing.Optional[HeadNodeConfig] = None
    enable_autoscaling: bool = False
    runtime_env: typing.Optional[dict] = None
    address: typing.Optional[str] = None
    shutdown_after_job_finishes: bool = False
    ttl_seconds_after_finished: typing.Optional[int] = None


@dataclass(kw_only=True)
class RayFunctionTask(AsyncFunctionTaskTemplate):
    """
    Actual Plugin that transforms the local python code for execution within Ray job.
    """

    task_type: str = "ray"
    plugin_config: RayJobConfig
    debuggable: bool = True
    supports_reuse_policy: typing.ClassVar[bool] = True

    async def pre(self, *args, **kwargs) -> Dict[str, Any]:
        init_params = {"address": self.plugin_config.address}

        if flyte.ctx().is_in_cluster():
            working_dir = os.getcwd()
            init_params["runtime_env"] = {
                "working_dir": working_dir,
                "excludes": ["script_mode.tar.gz", "fast*.tar.gz", ".python_history", ".code-server"],
            }

        if not ray.is_initialized():
            ray.init(**init_params)
        return {}

    def custom_config(self, sctx: SerializationContext) -> Optional[Dict[str, Any]]:
        cfg = self.plugin_config
        # Deprecated: runtime_env is removed KubeRay >= 1.1.0. It is replaced by runtime_env_yaml
        runtime_env = base64.b64encode(json.dumps(cfg.runtime_env).encode()).decode() if cfg.runtime_env else None
        runtime_env_yaml = yaml.dump(cfg.runtime_env) if cfg.runtime_env else None

        head_group_spec = None
        if cfg.head_node_config:
            head_pod_template = _build_node_pod_template(
                primary_container_name=_RAY_HEAD_CONTAINER_NAME,
                pod_template=cfg.head_node_config.pod_template,
                requests=cfg.head_node_config.requests,
                limits=cfg.head_node_config.limits,
            )

            head_group_spec = HeadGroupSpec(
                ray_start_params=cfg.head_node_config.ray_start_params,
                k8s_pod=head_pod_template.to_k8s_pod() if head_pod_template else None,
                extended_resources=get_proto_extended_resources(cfg.head_node_config.requests),
            )

        worker_group_spec: typing.List[WorkerGroupSpec] = []
        for c in cfg.worker_node_config:
            worker_pod_template = _build_node_pod_template(
                primary_container_name=_RAY_WORKER_CONTAINER_NAME,
                pod_template=c.pod_template,
                requests=c.requests,
                limits=c.limits,
            )

            worker_group_spec.append(
                WorkerGroupSpec(
                    group_name=c.group_name,
                    replicas=c.replicas,
                    min_replicas=c.min_replicas,
                    max_replicas=c.max_replicas,
                    ray_start_params=c.ray_start_params,
                    k8s_pod=worker_pod_template.to_k8s_pod() if worker_pod_template else None,
                    extended_resources=get_proto_extended_resources(c.requests),
                )
            )

        ray_job = RayJob(
            ray_cluster=RayCluster(
                head_group_spec=head_group_spec,
                worker_group_spec=worker_group_spec,
                enable_autoscaling=(cfg.enable_autoscaling or False),
            ),
            runtime_env=runtime_env,
            runtime_env_yaml=runtime_env_yaml,
            ttl_seconds_after_finished=cfg.ttl_seconds_after_finished,
            shutdown_after_job_finishes=cfg.shutdown_after_job_finishes,
        )

        custom = MessageToDict(ray_job)

        if self.reusable is not None:
            # `replicas` is the number of shared clusters; only 1 is supported for now.
            if self.reusable.max_replicas != 1:
                raise flyte.errors.RuntimeUserError(
                    "BadConfiguration",
                    f"Reusable Ray tasks currently support exactly 1 replica (one shared RayCluster); "
                    f"got replicas={self.reusable.replicas}. Use ReusePolicy(replicas=1).",
                )
            idle_ttl = self.reusable.idle_ttl
            scaledown_ttl = self.reusable.get_scaledown_ttl()
            custom["reusePolicy"] = {
                "parallelism": self.reusable.concurrency,
                "min_replica_count": self.reusable.min_replicas,
                "replica_count": self.reusable.max_replicas,
                "ttl_seconds": idle_ttl.total_seconds() if idle_ttl else None,  # type: ignore[union-attr]
                "scaledown_ttl_seconds": scaledown_ttl.total_seconds() if scaledown_ttl else None,
            }

        return custom


TaskPluginRegistry.register(config_type=RayJobConfig, plugin=RayFunctionTask)
