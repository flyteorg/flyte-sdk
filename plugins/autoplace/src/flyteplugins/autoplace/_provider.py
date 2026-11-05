"""
The seam between the cloud-neutral orchestration and one cloud's way of getting machines.

Everything in `_task.py`, `_race.py` and `_environment.py` (probe every cluster, first wins, release
the rest, place, retry on a missed hold) is the same on every cloud. What differs is answered by a
`CapacityProvider`: how to ask clusters for capacity and know it is there, how a pod claims it, and
how it is given back. `flyteplugins.autoplace.gke` is the first provider; an Azure Kubernetes Service one would be the
same ProvisioningRequest API with a different class name, an AWS one would need some other
mechanism, since Karpenter has no ask-first API.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Awaitable, Callable, Dict, List, Optional, Protocol, Sequence, Union

from flyte._pod import PodTemplate
from flyte._task_environment import TaskEnvironment

from ._shape import PodSet


@dataclass
class Reservation:
    """
    Outcome of one probe. `ok=False` is a normal result, not an error: a probe never raises for
    "no capacity", so a failed search is never replayed as a failure and the orchestration keeps the
    other clusters' answers. `name`/`namespace` are the provider's handle for the reserved capacity
    (for GKE, the ProvisioningRequest), which `consume` turns into pod metadata and `release` gives
    back. `provisioned_at` is when the cloud says the capacity came up (epoch seconds, 0 if unknown):
    the cloud's hold on the machines counts from then.
    """

    queue: str
    ok: bool = False
    name: str = ""
    namespace: str = ""
    reason: str = ""
    provisioned_at: float = 0.0


@dataclass
class Selection:
    """
    Outcome of one search across clusters: the reservation to place on, the other reservations that
    also came through (several clusters can provision at the same moment, and a probe that has already
    won cannot be cancelled) and must be released, and why each remaining cluster said no.

    `unresolved` are the probes that were cancelled before they reported. Cancelling hides the outcome:
    such a probe may still have reserved capacity (it can win in the moment the cancel is on its way),
    so each is listed with the handle it would have used, for the caller to release; releasing one that
    reserved nothing must be harmless.
    """

    winner: Optional[Reservation] = None
    extras: List[Reservation] = field(default_factory=list)
    misses: Dict[str, str] = field(default_factory=dict)
    unresolved: List[Reservation] = field(default_factory=list)


class CapacityProvider(Protocol):
    """
    What a cloud has to supply for AutoPlace to run on it:

    1. `select`: ask every candidate queue's cluster for the pod shapes at once and report which one
       got the capacity first (see `race`). It must be a *child action*, typically a task on one of
       `environments`, so that a retried driver replays the same answer instead of searching again,
       and its per-cluster probes must run in their clusters with those clusters' own credentials.
    2. `consume`: make the placed task's pods land on the reserved machines, given the reservation:
       annotations, node selectors, tolerations merged into the task's own pod template.
    3. `release`: give a reservation back once it is not (or no longer) needed: an extra winner right
       away, the placed winner after its task finishes. Also a child action, in the reservation's
       cluster. Releasing something already gone must succeed.

    `capacity_hold` is how long the cloud keeps the machines waiting for pods; the orchestration
    bounds the placed task's queue time under it and searches again on a miss. `environments` are
    deployed alongside the task environment. `validate` refuses shapes the cloud cannot provision.
    """

    name: str
    capacity_hold: timedelta

    @property
    def environments(self) -> Sequence[TaskEnvironment]: ...

    def validate(self, podsets: Sequence[PodSet]) -> None: ...

    def select(self, queues: Sequence[str], podsets: Sequence[PodSet], wait: timedelta) -> Awaitable[Selection]: ...

    def consume(self, pod_template: Optional[PodTemplate], reservation: Reservation) -> PodTemplate: ...

    def release(self, reservation: Reservation) -> Awaitable[None]: ...


ProviderFactory = Callable[[], CapacityProvider]

_REGISTRY: Dict[str, ProviderFactory] = {}
# Providers shipped with this package, imported on first use so `import flyteplugins.autoplace` stays
# light; each module registers itself on import.
_BUILTIN: Dict[str, str] = {"gke": "flyteplugins.autoplace.gke"}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Make `AutoPlaceEnvironment(provider=name)` build the provider with `factory()`."""
    _REGISTRY[name] = factory


def resolve_provider(provider: Union[str, CapacityProvider]) -> CapacityProvider:
    if not isinstance(provider, str):
        return provider
    if provider not in _REGISTRY and provider in _BUILTIN:
        importlib.import_module(_BUILTIN[provider])
    try:
        return _REGISTRY[provider]()
    except KeyError:
        known = sorted(set(_REGISTRY) | set(_BUILTIN))
        raise ValueError(f"Unknown capacity provider {provider!r}; known providers: {known}") from None
