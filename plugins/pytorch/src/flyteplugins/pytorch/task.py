import os
import typing
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import flyte
import torch
from flyte import PodTemplate, Resources
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
    rdzv_backend: str = "gloo"
    worker_node_config: typing.Optional[WorkerNodeConfig] = None
    master_node_config: typing.Optional[MasterNodeConfig] = None
    nnodes: Union[int, str] = 1
    nproc_per_node: int = 1
    run_policy: Optional[_kubeflow_common.RunPolicy] = None
    max_restarts: int = 3


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
    plugin_config: TorchJobConfig

    async def pre(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Pre execution setup for TorchFunctionTask.
        """

        if flyte.ctx().is_in_cluster():
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend=os.environ["BACKEND"],
                    init_method="env://",
                    rank=int(os.environ["RANK"]),
                    world_size=int(os.environ["WORLD_SIZE"]),
                )

        def wrapper(*args, **kwargs):
            if flyte.ctx().is_in_cluster():
                return torch.nn.parallel.DistributedDataParallel(*args, **kwargs)
            return args[0]

        return {
            "ddp_model": wrapper,
            "rank": int(os.environ.get("RANK", "0")),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        }

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
        if flyte.ctx().is_in_cluster():
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

        return return_vals

    def _convert_master_node_spec(self, config: MasterNodeConfig):
        return _convert_replica_spec(config, is_master=True)


TaskPluginRegistry.register(config_type=TorchJobConfig, plugin=TorchFunctionTask)
