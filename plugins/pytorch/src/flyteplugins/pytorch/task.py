import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Union

import flyte
from cloudpickle import cloudpickle
from flyte._logging import logger
from flyte._task import P, R
from flyte.extend import AsyncFunctionTaskTemplate, TaskPluginRegistry
from flyte.models import SerializationContext
from flyteidl.plugins.kubeflow import common_pb2
from flyteidl.plugins.kubeflow.pytorch_pb2 import (
    DistributedPyTorchTrainingReplicaSpec,
    DistributedPyTorchTrainingTask,
    ElasticConfig,
)
from google.protobuf.json_format import MessageToDict
from torch.distributed import run
from torch.distributed.launcher.api import LaunchConfig, elastic_launch


class CleanPodPolicy(Enum):
    NONE = common_pb2.CLEANPOD_POLICY_NONE
    ALL = common_pb2.CLEANPOD_POLICY_ALL
    RUNNING = common_pb2.CLEANPOD_POLICY_RUNNING


@dataclass
class RunPolicy:
    clean_pod_policy: CleanPodPolicy = None
    ttl_seconds_after_finished: Optional[int] = None
    active_deadline_seconds: Optional[int] = None
    backoff_limit: Optional[int] = None


@dataclass
class Elastic:
    rdzv_backend: str = "c10d"
    backend: str = "gloo"
    nnodes: Union[int, str] = 1
    nproc_per_node: int = 1
    run_policy: Optional[RunPolicy] = None
    monitor_interval: int = 3
    max_restarts: int = 3
    rdzv_configs: Dict[str, Any] = field(default_factory=lambda: {"timeout": 900, "join_timeout": 900})


def launcher_entrypoint(fn: bytes, kwargs: dict):
    fn = cloudpickle.loads(fn)
    return fn(**kwargs)


@dataclass(kw_only=True)
class TorchFunctionTask(AsyncFunctionTaskTemplate):
    """
    Plugin to transform local python code for execution as a PyTorch job.
    """

    task_type: str = "pytorch"
    task_type_version: int = 1
    plugin_config: Elastic

    def __post_init__(self):
        super().__post_init__()
        self.task_type = "python-task" if self.plugin_config.nnodes == 1 else "pytorch"
        self.min_nodes, self.max_nodes = run.parse_min_max_nnodes(str(self.plugin_config.nnodes))
        self.rdzv_backend = "c10d"

    async def pre(self, *args: P.args, **kwargs: P.kwargs) -> Dict[str, Any]:
        # If OMP_NUM_THREADS is not set, set it to 1 to avoid overloading the system.
        # Doing so to copy the default behavior of torchrun.
        # See https://github.com/pytorch/pytorch/blob/eea4ece256d74c6f25c1f4eab37b3f2f4aeefd4d/torch/distributed/run.py#L791
        if "OMP_NUM_THREADS" not in os.environ and self.plugin_config.nproc_per_node > 1:
            omp_num_threads = 1
            logger.warning(
                "\n*****************************************\n"
                "Setting OMP_NUM_THREADS environment variable for each process to be "
                "%s in default, to avoid your system being overloaded, "
                "please further tune the variable for optimal performance in "
                "your application as needed. \n"
                "*****************************************",
                omp_num_threads,
            )
            os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)
        return {}

    async def execute(self, *args: P.args, **kwargs: P.kwargs) -> R:
        config = LaunchConfig(
            run_id=flyte.ctx().action.run_name,
            min_nodes=self.min_nodes,
            max_nodes=self.max_nodes,
            nproc_per_node=self.plugin_config.nproc_per_node,
            rdzv_backend=self.rdzv_backend,
            rdzv_configs=self.plugin_config.rdzv_configs,
            rdzv_endpoint=os.environ.get("PET_RDZV_ENDPOINT", "localhost:0"),
            max_restarts=self.plugin_config.max_restarts,
            monitor_interval=self.plugin_config.monitor_interval,
        )
        out = elastic_launch(config=config, entrypoint=launcher_entrypoint)(cloudpickle.dumps(self.func), kwargs)

        # `out` is a dictionary of rank (not local rank) -> result
        # Rank 0 returns the result of the task function
        if 0 in out:
            return out[0]
        return None

    def custom_config(self, sctx: SerializationContext) -> Optional[Dict[str, Any]]:
        """
        Converts the ElasticConfig to a DistributedPyTorchTrainingTask
        """
        elastic_config = ElasticConfig(
            rdzv_backend=self.plugin_config.rdzv_backend,
            min_replicas=self.min_nodes,
            max_replicas=self.max_nodes,
            nproc_per_node=self.plugin_config.nproc_per_node,
            max_restarts=self.plugin_config.max_restarts,
        )

        policy = None
        if self.plugin_config.run_policy:
            policy = common_pb2.RunPolicy(
                clean_pod_policy=(
                    self.plugin_config.run_policy.clean_pod_policy.value
                    if self.plugin_config.run_policy.clean_pod_policy
                    else None
                ),
                ttl_seconds_after_finished=self.plugin_config.run_policy.ttl_seconds_after_finished,
                active_deadline_seconds=self.plugin_config.run_policy.active_deadline_seconds,
                backoff_limit=self.plugin_config.run_policy.backoff_limit,
            )

        torch_job = DistributedPyTorchTrainingTask(
            worker_replicas=DistributedPyTorchTrainingReplicaSpec(
                replicas=self.max_nodes,
            ),
            run_policy=policy,
            elastic_config=elastic_config,
        )

        return MessageToDict(torch_job)


TaskPluginRegistry.register(config_type=Elastic, plugin=TorchFunctionTask)
