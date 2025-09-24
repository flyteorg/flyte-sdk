import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Literal, Optional, Union

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
    """
    CleanPodPolicy defines how pods are cleaned up after a PyTorchJob completes.
    """

    NONE = common_pb2.CLEANPOD_POLICY_NONE
    ALL = common_pb2.CLEANPOD_POLICY_ALL
    RUNNING = common_pb2.CLEANPOD_POLICY_RUNNING


@dataclass
class RunPolicy:
    """
    RunPolicy describes some policy to apply to the execution of a kubeflow job.

    Args:
        clean_pod_policy (CleanPodPolicy): Defines the policy for cleaning up pods after
            the PyTorchJob completes. Defaults to None.
        ttl_seconds_after_finished (int, optional): Defines the TTL (in seconds) for cleaning
            up finished PyTorchJobs. Defaults to None.
        active_deadline_seconds (int, optional): Specifies the duration (in seconds) since
            startTime during which the job can remain active before it is terminated.
            Must be a positive integer. Applies only to pods where restartPolicy is
            OnFailure or Always. Defaults to None.
        backoff_limit (int, optional): Number of retries before marking this job as failed.
            Defaults to None.
    """

    clean_pod_policy: Optional[CleanPodPolicy] = None
    ttl_seconds_after_finished: Optional[int] = None
    active_deadline_seconds: Optional[int] = None
    backoff_limit: Optional[int] = None


@dataclass
class Elastic:
    """
    Elastic defines the configuration for running a PyTorch elastic job using torch.distributed.

    Args:
        nnodes (Union[int, str]): Number of nodes to use. Can be a fixed int or a range
            string (e.g., "2:4" for elastic training).
        nproc_per_node (int): Number of processes to launch per node.
        rdzv_backend (literal): Rendezvous backend to use. Typically "c10d". Defaults to "c10d".
        run_policy (RunPolicy, optional): Run policy applied to the job execution.
            Defaults to None.
        monitor_interval (int): Interval (in seconds) to monitor the job's state.
            Defaults to 3.
        max_restarts (int): Maximum number of worker group restarts before failing the job.
            Defaults to 3.
        rdzv_configs (Dict[str, Any]): Rendezvous configuration key-value pairs.
            Defaults to {"timeout": 900, "join_timeout": 900}.
    """

    nnodes: Union[int, str]
    nproc_per_node: int
    rdzv_backend: Literal["c10d", "etcd", "etcd-v2"] = "c10d"
    run_policy: Optional[RunPolicy] = None
    monitor_interval: int = 3
    max_restarts: int = 3
    rdzv_configs: Dict[str, Any] = field(default_factory=lambda: {"timeout": 900, "join_timeout": 900})


def launcher_entrypoint(fn: bytes, kwargs: dict):
    func = cloudpickle.loads(fn)
    return func(**kwargs)


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
            rdzv_backend=self.plugin_config.rdzv_backend,
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
