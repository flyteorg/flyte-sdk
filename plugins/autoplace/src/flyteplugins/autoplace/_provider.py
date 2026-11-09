"""
The seam between the cloud-neutral orchestration and one cloud's way of getting machines.

Everything in `_task.py` and `_environment.py` (probe every cluster, first wins, release the rest,
place) is the same on every cloud. What differs is answered by a `CapacityProvider`: how one cluster
is asked for capacity, how a pod claims it, and how it is given back. `flyteplugins.autoplace.gke` is
the first provider; an Azure Kubernetes Service one would be the same ProvisioningRequest API with a
different class name, an AWS one would need some other mechanism, since Karpenter has no ask-first API.
"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from datetime import timedelta
from typing import Awaitable, Callable, Dict, Optional, Protocol, Sequence, Union

from flyte import PodTemplate, TaskEnvironment

from ._shape import PodSet


@dataclass
class Reservation:
    """Capacity reserved in `queue`'s cluster under the provider's handle `name`."""

    queue: str
    name: str


def reservation_name(search: str, queue: str) -> str:
    """
    The handle a search's probe in `queue` reserves under: fixed by the search and the queue, so every
    queue but the winner's can be released by name without hearing back from its probe, and a retried
    parent asks again under the same names. A valid Kubernetes object name.
    """
    return f"autoplace-{hashlib.sha256(f'{search}/{queue}'.encode()).hexdigest()[:10]}"


class CapacityProvider(Protocol):
    """
    What a cloud has to supply for AutoPlace to run on it:

    1. `reserve`: ask one queue's cluster for the pod shapes, as a child task submitted to that queue
       (so it runs in the cluster, with its credentials, and a retried parent replays it). It returns
       once the capacity is there and raises otherwise; the search bounds it with `deadline`.
    2. `consume`: make the placed task's pods land on the reserved machines: annotations, node
       selectors, tolerations merged into the task's own pod template.
    3. `release`: give a reservation back, also a child task in the reservation's cluster. Releasing
       something that was never reserved, or is already gone, must succeed.

    `capacity_hold` is how long the cloud keeps reserved machines waiting for pods, which bounds the
    placed task's queue time; `max_run` is how long they live, which bounds its deadline.
    `environments` are deployed alongside the task environment. `validate` refuses shapes the cloud
    cannot provision.
    """

    name: str
    capacity_hold: timedelta
    max_run: timedelta

    @property
    def environments(self) -> Sequence[TaskEnvironment]: ...

    def validate(self, podsets: Sequence[PodSet]) -> None: ...

    def reserve(
        self, queue: str, name: str, podsets: Sequence[PodSet], deadline: timedelta, service_account: str
    ) -> Awaitable[Reservation]: ...

    def consume(self, pod_template: Optional[PodTemplate], reservation: Reservation) -> PodTemplate: ...

    def release(self, reservation: Reservation, service_account: str) -> Awaitable[None]: ...


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
