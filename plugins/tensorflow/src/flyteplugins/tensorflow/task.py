from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from cloudpickle import cloudpickle
from flyteidl.plugins.kubeflow import common_pb2
from flyteidl.plugins.kubeflow.tensorflow_pb2 import (
    DistributedTensorflowTrainingReplicaSpec,
    DistributedTensorflowTrainingTask,
)
from google.protobuf.json_format import MessageToDict

import flyte
from flyte._context import internal_ctx
from flyte.extend import AsyncFunctionTaskTemplate, TaskPluginRegistry
from flyte.models import SerializationContext, TaskContext


@dataclass
class RunPolicy:
    """
    RunPolicy describes some policy to apply to the execution of a kubeflow job.

    Args:
        clean_pod_policy (str, optional): Policy for cleaning up pods after the PyTorchJob completes.
            Allowed values are "None", "all", or "Running". Defaults to None.
        ttl_seconds_after_finished (int, optional): Defines the TTL (in seconds) for cleaning
            up finished PyTorchJobs. Defaults to None.
        active_deadline_seconds (int, optional): Specifies the duration (in seconds) since
            startTime during which the job can remain active before it is terminated.
            Must be a positive integer. Applies only to pods where restartPolicy is
            OnFailure or Always. Defaults to None.
        backoff_limit (int, optional): Number of retries before marking this job as failed.
            Defaults to None.
    """

    clean_pod_policy: Optional[Literal["None", "all", "Running"]] = None
    ttl_seconds_after_finished: Optional[int] = None
    active_deadline_seconds: Optional[int] = None
    backoff_limit: Optional[int] = None


@dataclass
class Tensorflow:
    """
    Tensorflow defines the configuration for running a Tensorflow job

    Args:
        run_policy (RunPolicy, optional): Run policy applied to the job execution.
            Defaults to None.
    """

    run_policy: Optional[RunPolicy] = None


def launcher_entrypoint(tctx: TaskContext, fn: bytes, kwargs: dict):
    func = cloudpickle.loads(fn)
    flyte.init(
        org=tctx.action.org,
        project=tctx.action.project,
        domain=tctx.action.domain,
        root_dir=tctx.run_base_dir,
    )

    with internal_ctx().replace_task_context(tctx):
        return func(**kwargs)


@dataclass(kw_only=True)
class TensorflowFunctionTask(AsyncFunctionTaskTemplate):
    """
    Plugin to transform local python code for execution as a Tensorflow job.
    """

    task_type: str = "tensorflow"
    task_type_version: int = 1
    plugin_config: Tensorflow

    def __post_init__(self):
        super().__post_init__()

    def custom_config(self, sctx: SerializationContext) -> Optional[Dict[str, Any]]:
        """
        Converts a DistributedTensorflowTrainingTask and then to a dictionary.
        """

        policy = None
        if self.plugin_config.run_policy:
            policy = common_pb2.RunPolicy(
                clean_pod_policy=(
                    # https://github.com/flyteorg/flyte/blob/4caa5639ee6453d86c823181083423549f08f702/flyteidl/protos/flyteidl/plugins/kubeflow/common.proto#L9-L13
                    common_pb2.CleanPodPolicy.Value(
                        "CLEANPOD_POLICY_" + self.plugin_config.run_policy.clean_pod_policy.upper()
                    )
                    if self.plugin_config.run_policy.clean_pod_policy
                    else None
                ),
                ttl_seconds_after_finished=self.plugin_config.run_policy.ttl_seconds_after_finished,
                active_deadline_seconds=self.plugin_config.run_policy.active_deadline_seconds,
                backoff_limit=self.plugin_config.run_policy.backoff_limit,
            )

        tensorflow_job = DistributedTensorflowTrainingTask(
            worker_replicas=DistributedTensorflowTrainingReplicaSpec(),
            run_policy=policy,
        )

        return MessageToDict(tensorflow_job)


TaskPluginRegistry.register(config_type=Tensorflow, plugin=TensorflowFunctionTask)
