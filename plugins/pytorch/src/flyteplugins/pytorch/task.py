import os
import typing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Union

import flyte
import torch
from cloudpickle import cloudpickle
from flyte import PodTemplate
from flyte._internal.runtime.resources_serde import get_proto_resources
from flyte._logging import logger
from flyte._task import P, R
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
from torch.distributed import run
from torch.distributed.launcher.api import LaunchConfig, elastic_launch


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
    worker_node_config: WorkerNodeConfig
    master_node_config: MasterNodeConfig
    rdzv_backend: str = "c10d"
    backend: str = "gloo"
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
    plugin_config: TorchJobConfig

    def __post_init__(self):
        super().__post_init__()
        self.min_nodes, self.max_nodes = run.parse_min_max_nnodes(str(self.plugin_config.nnodes))
        self.rdzv_backend = "c10d"

    async def pre(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Pre execution setup for TorchFunctionTask.
        """

    async def execute(self, *args: P.args, **kwargs: P.kwargs) -> R:
        dist_env_vars_set = os.environ.get("PET_NNODES") is not None
        if not dist_env_vars_set and self.min_nodes > 1:
            logger.warning(
                (
                    f"`nnodes` is set to {self.plugin_config.nnodes} in elastic task but execution appears "
                    "to not run in a `PyTorchJob`. Rendezvous might timeout."
                )
            )

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

        config = LaunchConfig(
            run_id=flyte.ctx().action.run_name,
            min_nodes=self.min_nodes,
            max_nodes=self.max_nodes,
            nproc_per_node=self.plugin_config.nproc_per_node,
            rdzv_backend=self.rdzv_backend,  # rdzv settings
            rdzv_configs=self.plugin_config.rdzv_configs,
            rdzv_endpoint=os.environ.get("PET_RDZV_ENDPOINT", "localhost:0"),
            max_restarts=self.plugin_config.max_restarts,
            monitor_interval=self.plugin_config.monitor_interval,
            start_method=self.plugin_config.start_method,
        )

        if self.plugin_config.start_method == "spawn":
            """
            We use cloudpickle to serialize the non-pickleable task function.
            The torch elastic launcher then launches the spawn_helper function (which is pickleable)
            instead of the task function. This helper function, in the child-process, then deserializes
            the task function, again with cloudpickle, and executes it.
            """

            async def launcher_target_func(fn: bytes, **kwargs):
                fn = cloudpickle.loads(fn)
                return await fn(**kwargs)

            dumped_target_function = cloudpickle.dumps(self.func)

            launcher_args = (
                dumped_target_function,
                kwargs,
            )
        elif self.plugin_config.start_method == "fork":
            """
            The torch elastic launcher doesn't support passing kwargs to the target function,
            only args. Flyte only works with kwargs. Thus, we create a closure which already has
            the task kwargs bound. We tell the torch elastic launcher to start this function in
            the child processes.
            """

            async def launcher_target_func():
                """Closure of the task function with kwargs already bound."""
                return await self.func(**kwargs)

            launcher_args = ()
        else:
            raise ValueError(f"Unsupported start method {self.plugin_config.start_method}")

        out = elastic_launch(config=config, entrypoint=launcher_target_func)(*launcher_args)

        # `out` is a dictionary of rank (not local rank) -> result
        # Rank 0 returns the result of the task function
        if 0 in out:
            return out[0].return_value
        return None

    def custom_config(self, sctx: SerializationContext) -> Any:
        """
        Converts the TorchJobConfig to a DistributedPyTorchTrainingTask
        """
        elastic_config = ElasticConfig(
            rdzv_backend=self.plugin_config.rdzv_backend,
            min_replicas=self.min_nodes,
            max_replicas=self.max_nodes,
            nproc_per_node=self.plugin_config.nproc_per_node,
            max_restarts=self.plugin_config.max_restarts,
        )

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
