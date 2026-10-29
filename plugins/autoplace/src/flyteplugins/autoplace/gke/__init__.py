"""
GKE capacity provider: Dynamic Workload Scheduler, flex-start with queued provisioning.

`reserve` runs `reserve_capacity` (a task on `probe_env`) in the target cluster; it files a GKE
ProvisioningRequest for the pod shapes and waits for `Provisioned=True`. `consume` adds the two GKE
annotations, the flex-start node selector and the queued-provisioning toleration to the placed task's
pod template. GKE holds provisioned nodes for 10 minutes before removing them if no pod consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Awaitable, ClassVar, Optional, Sequence

from flyte._pod import PodTemplate
from flyte._task_environment import TaskEnvironment

from .._provider import Reservation, register_provider
from .._shape import PodSet
from ._probe import evaluate_conditions, probe_env, reserve_capacity
from ._provreq import (
    ACCELERATOR_LABEL,
    CLASS_ANNOTATION,
    CONSUME_ANNOTATION,
    FLEX_START_LABEL,
    QUEUED_PROVISIONING_CLASS,
    QUEUED_TAINT,
    consume_pod_template,
)

MAX_RUN_CEILING = timedelta(days=7)


@dataclass(frozen=True)
class GKEFlexStart:
    """
    Args:
        max_run: `maxRunDurationSeconds` of the ProvisioningRequest: how long the provisioned nodes
            may live. GKE's default and ceiling is 7 days.
        accelerator: Explicit `cloud.google.com/gke-accelerator` value for the request's pod
            template. Defaults to the value the SDK serializes for the GPU name (the worker pod's
            selector); a name the SDK does not map leaves it unset and the autoscaler picks any
            flex-start pool whose nodes fit.

        probe_service_account: Kubernetes service account the probe runs as. Defaults to the probe
            environment's own (`autoplace`); set it when the RBAC is bound to another account.

    Per-cluster requirements: a flex-start node pool with queued provisioning for the accelerator
    (`--flex-start --enable-queued-provisioning`, min 0, location policy ANY; pre-created so the
    Union scheduler knows the pool's shape), and a `Role` in each project namespace letting the
    `autoplace` service account create/get/delete `provisioningrequests.autoscaling.x-k8s.io`,
    create/delete `podtemplates` and get `pods`.
    """

    name: ClassVar[str] = "gke"
    capacity_hold: ClassVar[timedelta] = timedelta(minutes=10)

    max_run: timedelta = MAX_RUN_CEILING
    accelerator: Optional[str] = None
    probe_service_account: Optional[str] = None

    def __post_init__(self) -> None:
        if not timedelta(0) < self.max_run <= MAX_RUN_CEILING:
            raise ValueError(f"max_run must be within (0, {MAX_RUN_CEILING}]; GKE caps flex-start at 7 days")

    @property
    def environments(self) -> Sequence[TaskEnvironment]:
        return (probe_env,)

    def validate(self, podsets: Sequence[PodSet]) -> None:
        # Flex-start also provisions TPUs, but a TPU shape (topology, slice) is not expressible as a
        # pod request, so it is left for a later version.
        for ps in podsets:
            if ps.gpu_count and ps.gpu_class != "GPU":
                raise ValueError(
                    f"The GKE flex-start provider supports NVIDIA GPUs only; got device class {ps.gpu_class!r}"
                )

    def reserve(self, queue: str, podsets: Sequence[PodSet], wait: timedelta) -> Awaitable[Reservation]:
        # `override` is per call, so this takes effect where mutating `probe_env` would not: the template
        # copied the environment's service account when `reserve_capacity` was decorated.
        probe = reserve_capacity.override(queue=queue, service_account=self.probe_service_account)
        return probe(
            queue, list(podsets), int(self.max_run.total_seconds()), int(wait.total_seconds()), self.accelerator
        )

    def consume(self, pod_template: Optional[PodTemplate], reservation: Reservation) -> PodTemplate:
        return consume_pod_template(pod_template, reservation.name)


register_provider(GKEFlexStart.name, GKEFlexStart)

__all__ = [
    "ACCELERATOR_LABEL",
    "CLASS_ANNOTATION",
    "CONSUME_ANNOTATION",
    "FLEX_START_LABEL",
    "MAX_RUN_CEILING",
    "QUEUED_PROVISIONING_CLASS",
    "QUEUED_TAINT",
    "GKEFlexStart",
    "consume_pod_template",
    "evaluate_conditions",
    "probe_env",
    "reserve_capacity",
]
