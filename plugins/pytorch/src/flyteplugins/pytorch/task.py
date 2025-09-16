import os
import typing
from dataclasses import dataclass, field
from enum import Enum
from functools import partial, update_wrapper
from typing import Any, Dict, Optional, Union

import torch
from flyte import PodTemplate
from flyte._internal.runtime.resources_serde import get_proto_resources
from flyte.extend import AsyncFunctionTaskTemplate, TaskPluginRegistry
from flyte.models import SerializationContext
from flyteidl.plugins import common_pb2 as _common_pb2
from flyteidl.plugins.kubeflow import common_pb2 as _kubeflow_common
from flyteidl.plugins.kubeflow.pytorch_pb2 import (
    DistributedPyTorchTrainingReplicaSpec,
    DistributedPyTorchTrainingTask,
    ElasticConfig,
)
from google.protobuf.json_format import MessageToDict
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

if typing.TYPE_CHECKING:
    pass

TORCH_IMPORT_ERROR_MESSAGE = "PyTorch is not installed"


class RestartPolicy(Enum):
    ALWAYS = _kubeflow_common.RESTART_POLICY_ALWAYS
    FAILURE = _kubeflow_common.RESTART_POLICY_ON_FAILURE
    NEVER = _kubeflow_common.RESTART_POLICY_NEVER


class CleanPodPolicy(Enum):
    NONE = _kubeflow_common.CLEANPOD_POLICY_NONE
    ALL = _kubeflow_common.CLEANPOD_POLICY_ALL
    RUNNING = _kubeflow_common.CLEANPOD_POLICY_RUNNING


@dataclass
class RunPolicy:
    clean_pod_policy: CleanPodPolicy = None
    ttl_seconds_after_finished: Optional[int] = None
    active_deadline_seconds: Optional[int] = None
    backoff_limit: Optional[int] = None


@dataclass
class MasterNodeConfig:
    image: Optional[str] = None
    replicas: Optional[int] = 1
    pod_template: typing.Optional[PodTemplate] = None


@dataclass
class WorkerNodeConfig:
    image: Optional[str] = None
    replicas: Optional[int] = None
    restart_policy: Optional[_common_pb2.RestartPolicy] = None


@dataclass
class TorchJobConfig:
    rdzv_backend: str = "c10d"
    backend: str = "gloo"
    worker_node_config: typing.Optional[WorkerNodeConfig] = None
    master_node_config: typing.Optional[MasterNodeConfig] = None
    nnodes: Union[int, str] = 1
    nproc_per_node: int = 1
    run_policy: Optional[RunPolicy] = None
    monitor_interval: int = 3
    max_restarts: int = 3
    rdzv_configs: Dict[str, Any] = field(default_factory=lambda: {"timeout": 900, "join_timeout": 900})
    start_method: str = "spawn"


def _distributed_entrypoint(user_entrypoint, backend="gloo", *args, **kwargs):
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend)

    return user_entrypoint(*args, **kwargs)


def _convert_run_policy_to_flyte_idl(
    run_policy: RunPolicy,
) -> _kubeflow_common.RunPolicy:
    return _kubeflow_common.RunPolicy(
        clean_pod_policy=(run_policy.clean_pod_policy.value if run_policy.clean_pod_policy else None),
        ttl_seconds_after_finished=run_policy.ttl_seconds_after_finished,
        active_deadline_seconds=run_policy.active_deadline_seconds,
        backoff_limit=run_policy.backoff_limit,
    )


@dataclass(kw_only=True)
class TorchFunctionTask(AsyncFunctionTaskTemplate):
    """
    Plugin to transform local python code for execution as a PyTorch job.
    """

    task_type: str = "torch"
    plugin_config: TorchJobConfig = field(default_factory=TorchJobConfig)

    async def pre(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Pre execution setup for TorchFunctionTask.
        """

        try:
            from torch.distributed import run
        except ImportError:
            raise ImportError(TORCH_IMPORT_ERROR_MESSAGE)

        min_nodes, max_nodes = run.parse_min_max_nnodes(str(self.plugin_config.nnodes))

        if self.plugin_config is None:
            raise ValueError("Plugin config must be provided for TorchFunctionTask")

        config = LaunchConfig(
            run_id=self.name,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            nproc_per_node=self.plugin_config.nproc_per_node,
            rdzv_backend=self.plugin_config.rdzv_backend,
            rdzv_configs=self.plugin_config.rdzv_configs,
            rdzv_endpoint=os.environ.get("PET_RDZV_ENDPOINT", "localhost:29500"),
            max_restarts=self.plugin_config.max_restarts,
            monitor_interval=self.plugin_config.monitor_interval,
            start_method=self.plugin_config.start_method,
        )

        def wrapper(entrypoint, *a, **kw):
            wrapped = partial(_distributed_entrypoint, entrypoint, backend=self.plugin_config.backend)
            update_wrapper(wrapped, entrypoint)

            return elastic_launch(config=config, entrypoint=wrapped)(*a, **kw)

        return {"elastic_launcher": wrapper}

    def custom_config(self, sctx: SerializationContext) -> Any:
        """
        Converts the TorchJobConfig to a DistributedPyTorchTrainingTask
        """

        try:
            from torch.distributed import run
        except ImportError:
            raise ImportError(TORCH_IMPORT_ERROR_MESSAGE)

        min_nodes, max_nodes = run.parse_min_max_nnodes(str(self.plugin_config.nnodes))

        elastic_config = ElasticConfig(
            rdzv_backend=self.plugin_config.rdzv_backend,
            min_replicas=min_nodes,
            max_replicas=max_nodes,
            nproc_per_node=self.plugin_config.nproc_per_node,
            max_restarts=self.plugin_config.max_restarts,
        )

        if self.plugin_config.master_node_config is None:
            raise ValueError("Master node configuration must be provided")

        if self.plugin_config.worker_node_config is None:
            raise ValueError("Worker node configuration must be provided")

        policy = (
            _convert_run_policy_to_flyte_idl(self.plugin_config.run_policy) if self.plugin_config.run_policy else None
        )

        torch_job = DistributedPyTorchTrainingTask(
            worker_replicas=self._create_worker_spec(),
            master_replicas=self._create_master_spec(),
            run_policy=policy,
            elastic_config=elastic_config,
        )

        return MessageToDict(torch_job)

    def _create_worker_spec(self) -> DistributedPyTorchTrainingReplicaSpec:
        if self.plugin_config.worker_node_config is None:
            raise ValueError("Worker node configuration must be provided")

        if self.image is None:
            raise ValueError("Either task image or worker node image must be provided")

        return DistributedPyTorchTrainingReplicaSpec(
            replicas=self.plugin_config.worker_node_config.replicas,
            image=str(self.image),
            resources=get_proto_resources(self.resources),
            restart_policy=_common_pb2.RestartPolicy.RESTART_POLICY_ON_FAILURE,
        )

    def _create_master_spec(self) -> DistributedPyTorchTrainingReplicaSpec:
        if self.plugin_config.master_node_config is None:
            raise ValueError("Master node configuration must be provided")

        if self.image is None:
            raise ValueError("Either task image or master node image must be provided")

        return DistributedPyTorchTrainingReplicaSpec(
            replicas=1,
            image=str(self.image),
            resources=get_proto_resources(self.resources),
            restart_policy=_common_pb2.RestartPolicy.RESTART_POLICY_ON_FAILURE,
        )

    async def post(self, return_vals: Any) -> Any:
        """
        Post execution cleanup for TorchFunctionTask.
        Make sure to destroy the process group if we are in a cluster environment.
        """
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        return return_vals


TaskPluginRegistry.register(config_type=TorchJobConfig, plugin=TorchFunctionTask)
