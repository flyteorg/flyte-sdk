from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Literal, Optional, Union

from flyte._task_environment import TaskEnvironment

if TYPE_CHECKING:
    pass


@dataclass(frozen=True, kw_only=True)
class TorchRun:
    """TorchRun launcher configuration for a ClusteredTaskEnvironment.

    :param rdzv_backend: Rendezvous backend. "static" (default) relies on JobSet-level restarts;
        "c10d" enables in-job elastic recovery via a TCPStore on rank-0.
    :param max_restarts: In-pod torchrun restarts before the pod itself fails. Distinct from
        JobSet-level max_restarts on ClusterFailurePolicy.
    """

    rdzv_backend: Literal["static", "c10d"] = "static"
    max_restarts: int = 0
    # master_port is intentionally absent — hardcoded to 29500 in the Go plugin


Runtime = Union[TorchRun]


@dataclass(frozen=True, kw_only=True)
class ClusterFailurePolicy:
    """Failure and restart policy for the JobSet as a whole.

    :param max_restarts: Number of times the entire JobSet may be restarted before Flyte
        surfaces a RetryableFailure.
    :param restart_on_host_maintenance: When True, node evictions (DisruptionTarget condition)
        trigger a free restart that does not consume the max_restarts budget.
    """

    max_restarts: int = 0
    restart_on_host_maintenance: bool = False


_INTERCONNECT_VALUES = ("tcp", "efa", "infiniband", "roce")


@dataclass(kw_only=True)
class ClusteredTaskEnvironment(TaskEnvironment):
    """A TaskEnvironment that emits a Kubernetes JobSet for distributed multi-node training.

    Inherits all fields from TaskEnvironment (name, image, resources, env_vars, secrets,
    pod_template, queue, cache, reusable). The fields below are specific to clustered execution.

    :param replicas: Number of pods (== number of nodes). Required.
    :param nproc_per_node: Number of processes per pod, passed to ``torchrun --nproc-per-node``.
        Must be >= 1 and, when resources.gpu is set, <= resources.gpu. Required.
    :param runtime: Launcher configuration. Phase 1 supports only TorchRun().
    :param interconnect: Network fabric. Non-TCP options require RDMA-capable images (Phase 3).
    :param failure_policy: JobSet-level restart and eviction policy.
    :param ttl_seconds_after_finished: Seconds to retain the JobSet after completion.
    """

    replicas: int
    nproc_per_node: int
    runtime: Runtime = field(default_factory=TorchRun)
    interconnect: Literal["tcp", "efa", "infiniband", "roce"] = "tcp"
    failure_policy: ClusterFailurePolicy = field(default_factory=ClusterFailurePolicy)
    ttl_seconds_after_finished: Optional[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.replicas < 1:
            raise ValueError("replicas must be >= 1")
        if self.nproc_per_node < 1:
            raise ValueError("nproc_per_node must be >= 1")
        if self.resources is not None and self.resources.gpu is not None:
            # get_device() normalizes int, "MODEL:N" strings, and GPU/TPU Device objects.
            device = self.resources.get_device()
            gpu_count = device.quantity if device is not None else None
            if gpu_count is not None and gpu_count < self.nproc_per_node:
                raise ValueError(f"resources.gpu ({gpu_count}) must be >= nproc_per_node ({self.nproc_per_node})")
        if not isinstance(self.runtime, TorchRun):
            raise TypeError(f"unsupported runtime type: {type(self.runtime).__name__}")
        if self.interconnect not in _INTERCONNECT_VALUES:
            raise ValueError(f"interconnect must be one of {_INTERCONNECT_VALUES}")
        if self.interconnect != "tcp":
            warnings.warn(
                f"interconnect={self.interconnect!r} requires RDMA-capable images (Phase 3)",
                stacklevel=2,
            )

    def to_custom_dict(self) -> Dict:
        """Serialize this environment to the dict shape expected by ClusteredTaskSpec proto.

        Imported lazily so the heavy clustered_pb2 module is only loaded at serialization
        time rather than on every ``flyte.distributed`` import.
        """
        from flyteidl2.plugins.clustered_pb2 import (
            ClusteredTaskSpec,
            Interconnect,
            RdzvBackend,
            TorchRuntime,
        )
        from flyteidl2.plugins.clustered_pb2 import (
            ClusterFailurePolicy as ClusteredFailurePolicyProto,
        )
        from flyteidl2.plugins.clustered_pb2 import (
            Runtime as RuntimeProto,
        )
        from google.protobuf.json_format import MessageToDict

        _rdzv_map = {"static": RdzvBackend.STATIC, "c10d": RdzvBackend.C10D}
        _interconnect_map = {
            "tcp": Interconnect.TCP,
            "efa": Interconnect.EFA,
            "infiniband": Interconnect.INFINIBAND,
            "roce": Interconnect.ROCE,
        }

        torch_runtime = TorchRuntime(
            rdzv_backend=_rdzv_map[self.runtime.rdzv_backend],
            max_restarts=self.runtime.max_restarts,
        )
        failure_policy = ClusteredFailurePolicyProto(
            max_restarts=self.failure_policy.max_restarts,
            restart_on_host_maintenance=self.failure_policy.restart_on_host_maintenance,
        )
        spec = ClusteredTaskSpec(
            replicas=self.replicas,
            nproc_per_node=self.nproc_per_node,
            runtime=RuntimeProto(torchrun=torch_runtime),
            interconnect=_interconnect_map[self.interconnect],
            failure_policy=failure_policy,
        )
        if self.ttl_seconds_after_finished is not None:
            spec.ttl_seconds_after_finished.value = self.ttl_seconds_after_finished
        return MessageToDict(spec)
