import os
import typing
from dataclasses import dataclass, field
from functools import partial, update_wrapper
from typing import Any, Dict, Optional, Union

import torch
from flyte import PodTemplate, Resources
from flyte._context import internal_ctx
from flyte.extend import AsyncFunctionTaskTemplate, TaskPluginRegistry
from flyte.models import SerializationContext
from flyteidl.core import tasks_pb2 as _tasks_pb2
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


@dataclass
class MasterNodeConfig:
    image: Optional[str] = None
    replicas: Optional[int] = 1
    pod_template: typing.Optional[PodTemplate] = None
    requests: Optional[Resources] = None
    limits: Optional[Resources] = None


@dataclass
class WorkerNodeConfig:
    image: Optional[str] = None
    requests: Optional[Resources] = None
    limits: Optional[Resources] = None
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
    run_policy: Optional[_kubeflow_common.RunPolicy] = None
    monitor_interval: int = 3
    max_restarts: int = 3
    rdzv_configs: Dict[str, Any] = field(default_factory=lambda: {"timeout": 900, "join_timeout": 900})
    start_method: str = "spawn"


def _distributed_entrypoint(user_entrypoint, backend="gloo", *args, **kwargs):
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend)

    return user_entrypoint(*args, **kwargs)


def _convert_replica_spec(replica_config, is_master: bool = False):
    if replica_config is None:
        raise ValueError("Replica configuration must be provided")

    return DistributedPyTorchTrainingReplicaSpec(
        replicas=replica_config.replicas if not is_master else 1,
        image=replica_config.image,
        resources=_tasks_pb2.Resources(
            requests=replica_config.requests,
            limits=replica_config.limits,
        ),
        restart_policy=_common_pb2.RestartPolicy.RESTART_POLICY_ON_FAILURE,
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
            run_id=os.environ.get("PET_RUN_ID", "flyte-pytorch-run"),
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

            ctx = internal_ctx()

            if ctx.data.task_context is None:
                raise ValueError("Task context is not available in the current context")

            config.run_id = ctx.data.task_context.action.name

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

        worker_spec = _convert_replica_spec(self.plugin_config.worker_node_config, is_master=False)
        master_spec = self._convert_master_node_spec(self.plugin_config.master_node_config)

        torch_job = DistributedPyTorchTrainingTask(
            worker_replicas=worker_spec,
            master_replicas=master_spec,
            run_policy=self.plugin_config.run_policy,
            elastic_config=elastic_config,
        )

        return MessageToDict(torch_job)

    async def post(self, return_vals: Any) -> Any:
        """
        Post execution cleanup for TorchFunctionTask.
        Make sure to destroy the process group if we are in a cluster environment.
        """
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        return return_vals

    def _convert_master_node_spec(self, config: MasterNodeConfig):
        return _convert_replica_spec(config, is_master=True)


TaskPluginRegistry.register(config_type=TorchJobConfig, plugin=TorchFunctionTask)
