import base64
import json
import os
import typing
from dataclasses import dataclass
from typing import Any, Dict, Optional

import flyte
import yaml
from flyte import PodTemplate, Resources
from flyte.extend import AsyncFunctionTaskTemplate, TaskPluginRegistry, pod_spec_from_resources
from flyte.models import SerializationContext
from flyteidl2.plugins.ray_pb2 import HeadGroupSpec, RayCluster, RayJob, WorkerGroupSpec
from google.protobuf.json_format import MessageToDict

import ray

if typing.TYPE_CHECKING:
    pass


_RAY_HEAD_CONTAINER_NAME = "ray-head"
_RAY_WORKER_CONTAINER_NAME = "ray-worker"


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
            if cfg.head_node_config.requests or cfg.head_node_config.limits:
                head_pod_template = PodTemplate(
                    pod_spec=pod_spec_from_resources(
                        primary_container_name=_RAY_HEAD_CONTAINER_NAME,
                        requests=cfg.head_node_config.requests,
                        limits=cfg.head_node_config.limits,
                    )
                )
            else:
                head_pod_template = cfg.head_node_config.pod_template

            head_group_spec = HeadGroupSpec(
                ray_start_params=cfg.head_node_config.ray_start_params,
                k8s_pod=head_pod_template.to_k8s_pod() if head_pod_template else None,
            )

        worker_group_spec: typing.List[WorkerGroupSpec] = []
        for c in cfg.worker_node_config:
            if c.requests or c.limits:
                worker_pod_template = PodTemplate(
                    pod_spec=pod_spec_from_resources(
                        primary_container_name=_RAY_WORKER_CONTAINER_NAME,
                        requests=c.requests,
                        limits=c.limits,
                    )
                )
            else:
                worker_pod_template = c.pod_template

            worker_group_spec.append(
                WorkerGroupSpec(
                    group_name=c.group_name,
                    replicas=c.replicas,
                    min_replicas=c.min_replicas,
                    max_replicas=c.max_replicas,
                    ray_start_params=c.ray_start_params,
                    k8s_pod=worker_pod_template.to_k8s_pod() if worker_pod_template else None,
                )
            )

        ray_job = RayJob(
            ray_cluster=RayCluster(
                head_group_spec=head_group_spec,
                worker_group_spec=worker_group_spec,
                enable_autoscaling=(cfg.enable_autoscaling if cfg.enable_autoscaling else False),
            ),
            runtime_env=runtime_env,
            runtime_env_yaml=runtime_env_yaml,
            ttl_seconds_after_finished=cfg.ttl_seconds_after_finished,
            shutdown_after_job_finishes=cfg.shutdown_after_job_finishes,
        )

        return MessageToDict(ray_job)


TaskPluginRegistry.register(config_type=RayJobConfig, plugin=RayFunctionTask)
