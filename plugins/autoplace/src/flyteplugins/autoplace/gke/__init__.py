"""
GKE capacity provider: Dynamic Workload Scheduler, flex-start with queued provisioning.

`reserve` runs `reserve_capacity` (a task on `probe_env`) in one candidate cluster; it files a GKE
ProvisioningRequest for the pod shapes and returns at `Provisioned=True`. `consume` adds the two GKE
annotations, the flex-start node selector and the queued-provisioning toleration to the placed task's
pod template. `release` runs `release_capacity` in the reservation's cluster, deleting the request. GKE
holds provisioned nodes for 10 minutes before removing them if no pod consumes them, and removes them
after `max_run` regardless.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Awaitable, ClassVar, Optional, Sequence

from flyte import PodTemplate, TaskEnvironment, Timeout

from .._provider import Reservation, register_provider
from .._shape import PodSet
from ._probe import CapacityRequestFailed, evaluate_conditions, probe_env, release_capacity, reserve_capacity
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
            may live, and so the placed task's deadline. GKE's default and ceiling is 7 days.
        accelerator: Explicit `cloud.google.com/gke-accelerator` value for the request's pod
            template. Defaults to the value the SDK serializes for the GPU name (the worker pod's
            selector); a name the SDK does not map leaves it unset and the autoscaler picks any
            flex-start pool whose nodes fit.

    Per-cluster requirements: a flex-start node pool with queued provisioning for the accelerator
    (`--flex-start --enable-queued-provisioning`, min 0, location policy ANY; pre-created rather than
    left to node auto-provisioning, so the pool exists before a request arrives), and a `Role` in each
    project namespace letting the environment's `service_account` create/get/patch/delete
    `provisioningrequests.autoscaling.x-k8s.io`, create/patch/delete `podtemplates` and get `pods`.
    Without `patch` a won request is not handed over and is deleted with its probe's pod.
    """

    name: ClassVar[str] = "gke"
    capacity_hold: ClassVar[timedelta] = timedelta(minutes=10)

    max_run: timedelta = MAX_RUN_CEILING
    accelerator: Optional[str] = None

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

    async def reserve(
        self, queue: str, name: str, podsets: Sequence[PodSet], deadline: timedelta, service_account: str
    ) -> Reservation:
        task = reserve_capacity.override(
            queue=queue, service_account=service_account, timeout=Timeout(deadline=deadline)
        )
        await task.aio(name, list(podsets), int(self.max_run.total_seconds()), self.accelerator)
        return Reservation(queue=queue, name=name)

    def release(self, reservation: Reservation, service_account: str) -> Awaitable[None]:
        task = release_capacity.override(queue=reservation.queue, service_account=service_account)
        return task.aio(reservation.name)

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
    "CapacityRequestFailed",
    "GKEFlexStart",
    "consume_pod_template",
    "evaluate_conditions",
    "probe_env",
    "release_capacity",
    "reserve_capacity",
]
