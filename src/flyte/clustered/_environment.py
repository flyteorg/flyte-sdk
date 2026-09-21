from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Literal, Optional, Union

from flyte._task_environment import TaskEnvironment

if TYPE_CHECKING:
    pass


_RDZV_BACKEND_VALUES = ("static", "c10d")


def _check_max_restarts(value: object, where: str) -> None:
    """Both `max_restarts` fields reach an int32 proto field untouched.

    A non-int dies there as an anonymous `TypeError: 'str' object cannot be interpreted as an
    integer`, and -- because the field is a signed int32 rather than the uint32 wrapper used for
    `ttl_seconds_after_finished` -- a negative one serializes *silently* into a restart budget the
    backend cannot honour.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{where} max_restarts must be an int, got {value!r}")
    if value < 0:
        raise ValueError(f"{where} max_restarts must be >= 0, got {value}")


@dataclass(frozen=True, kw_only=True)
class TorchRun:
    """TorchRun launcher configuration for a ClusteredTaskEnvironment.

    Args:
        rdzv_backend: Rendezvous backend. "static" (default) relies on JobSet-level restarts;
            "c10d" enables in-job elastic recovery via a TCPStore on rank-0.
        max_restarts: In-pod torchrun restarts before the pod itself fails. Distinct from
            JobSet-level max_restarts on ClusterFailurePolicy.
    """

    rdzv_backend: Literal["static", "c10d"] = "static"
    max_restarts: int = 0
    # master_port is intentionally absent — hardcoded to 29500 in the Go plugin

    def __post_init__(self) -> None:
        # `to_custom_dict` resolves this through `_rdzv_map[...]`, exactly as it resolves
        # `interconnect` through `_interconnect_map[...]`. `interconnect` is checked against
        # `_INTERCONNECT_VALUES` in `ClusteredTaskEnvironment.__post_init__`; this one was not, so a
        # near miss such as "C10D" surfaced as a bare `KeyError: 'C10D'` naming neither the field nor
        # the class. A `Literal` is not enforced at runtime, so the check has to be explicit.
        if self.rdzv_backend not in _RDZV_BACKEND_VALUES:
            raise ValueError(f"rdzv_backend must be one of {_RDZV_BACKEND_VALUES}, got {self.rdzv_backend!r}")
        _check_max_restarts(self.max_restarts, "TorchRun")


@dataclass(frozen=True, kw_only=True)
class JaxRun:
    """JAX multi-process runtime for a ClusteredTaskEnvironment.

    Each pod runs exactly one Python process (`nproc_per_node` must be 1) that owns every local
    accelerator — JAX's recommended multi-host layout. The pod-0 process hosts the `jax.distributed`
    coordinator on `MASTER_ADDR:MASTER_PORT`; every process must call `flyte.clustered.jax_initialize`
    before any JAX computation. No launcher binary is involved: the `clustered` entrypoint exports the
    process topology and execs `a0` directly.
    """


Runtime = Union[TorchRun, JaxRun]


def launcher_name(runtime: Runtime) -> str:
    """Name of the launcher the `clustered` entrypoint execs for `runtime` (its `--runtime` value)."""
    if isinstance(runtime, JaxRun):
        return "jax"
    return "torchrun"


@dataclass(frozen=True, kw_only=True)
class ClusterFailurePolicy:
    """Failure and restart policy for the JobSet as a whole.

    Args:
        max_restarts: Number of times the entire JobSet may be restarted before Flyte
            surfaces a RetryableFailure.
        restart_on_host_maintenance: When True, node evictions (DisruptionTarget condition)
            trigger a free restart that does not consume the max_restarts budget. Free restarts
            still increment `flyte.ctx().restart_attempt`.
    """

    max_restarts: int = 0
    restart_on_host_maintenance: bool = False

    def __post_init__(self) -> None:
        _check_max_restarts(self.max_restarts, "ClusterFailurePolicy")


_INTERCONNECT_VALUES = ("tcp",)


@dataclass(kw_only=True)
class ClusteredTaskEnvironment(TaskEnvironment):
    """A TaskEnvironment that emits a Kubernetes JobSet for distributed multi-node training.

    Inherits all fields from TaskEnvironment (name, image, resources, env_vars, secrets,
    pod_template, queue, cache, reusable). The fields below are specific to clustered execution.

    Args:
        replicas: Number of pods (== number of nodes). Required.
        nproc_per_node: Number of processes per pod. For TorchRun it is passed as
            `torchrun --nproc-per-node` (typically one per GPU); must be >= 1 and, when resources.gpu
            is set, <= resources.gpu. JaxRun runs one process per pod, so it must be 1. Required.
        runtime: Launcher configuration: TorchRun() (default) or JaxRun().
        interconnect: Network fabric. Currently only "tcp" is supported.
        failure_policy: JobSet-level restart and eviction policy.
        ttl_seconds_after_finished: Seconds to retain the JobSet after completion.
    """

    replicas: int
    nproc_per_node: int
    runtime: Runtime = field(default_factory=TorchRun)
    interconnect: Literal["tcp"] = "tcp"
    failure_policy: ClusterFailurePolicy = field(default_factory=ClusterFailurePolicy)
    ttl_seconds_after_finished: Optional[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.reusable is not None:
            raise ValueError("ClusteredTaskEnvironment does not support reusable environments")
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
        if not isinstance(self.runtime, (TorchRun, JaxRun)):
            raise TypeError(f"unsupported runtime type: {type(self.runtime).__name__}")
        if isinstance(self.runtime, JaxRun) and self.nproc_per_node != 1:
            raise ValueError(
                "JaxRun runs one process per pod that owns all local devices, so nproc_per_node must be 1 "
                f"(got {self.nproc_per_node}); restrict devices per process with "
                "jax_initialize(local_device_ids=...) instead"
            )
        if self.interconnect not in _INTERCONNECT_VALUES:
            raise ValueError(f"interconnect must be one of {_INTERCONNECT_VALUES}")
        # Unlike the two `max_restarts` fields, this one reaches a uint32 wrapper, so protobuf does
        # reject a negative -- but as `ValueError: Value out of range: -5`, which names no field.
        if self.ttl_seconds_after_finished is not None:
            if not isinstance(self.ttl_seconds_after_finished, int) or isinstance(
                self.ttl_seconds_after_finished, bool
            ):
                raise ValueError(
                    f"ttl_seconds_after_finished must be an int or None, got {self.ttl_seconds_after_finished!r}"
                )
            if self.ttl_seconds_after_finished < 0:
                raise ValueError(f"ttl_seconds_after_finished must be >= 0, got {self.ttl_seconds_after_finished}")
        # Route tasks built by this env to ClusteredTaskTemplate via the plugin registry.
        # Imported lazily to keep `import flyte.clustered` light (flyte.extend pulls in heavy deps).
        from flyte.clustered._task import _ClusteredPlugin

        self.plugin_config = _ClusteredPlugin()

    def to_custom_dict(self) -> Dict:
        """Serialize this environment to the dict shape expected by ClusteredTaskSpec proto.

        Imported lazily so the heavy clustered_pb2 module is only loaded at serialization
        time rather than on every `flyte.clustered` import.
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
        }

        failure_policy = ClusteredFailurePolicyProto(
            max_restarts=self.failure_policy.max_restarts,
            restart_on_host_maintenance=self.failure_policy.restart_on_host_maintenance,
        )
        spec = ClusteredTaskSpec(
            replicas=self.replicas,
            nproc_per_node=self.nproc_per_node,
            interconnect=_interconnect_map[self.interconnect],
            failure_policy=failure_policy,
        )
        if isinstance(self.runtime, TorchRun):
            spec.runtime.CopyFrom(
                RuntimeProto(
                    torchrun=TorchRuntime(
                        rdzv_backend=_rdzv_map[self.runtime.rdzv_backend],
                        max_restarts=self.runtime.max_restarts,
                    )
                )
            )
        # JaxRun: the `Runtime` oneof in the pinned flyteidl2 only has `torchrun`, and the backend reads
        # nothing runtime-specific from the spec (the JobSet env it injects is shared by every runtime).
        # The launcher is selected through the container args instead (`clustered --runtime=jax`, see
        # ClusteredTaskTemplate.container_args), so leave `runtime` unset rather than claim torchrun.
        # Adding a `jax` oneof variant is a backend follow-up.
        if self.ttl_seconds_after_finished is not None:
            spec.ttl_seconds_after_finished.value = self.ttl_seconds_after_finished
        return MessageToDict(spec)
