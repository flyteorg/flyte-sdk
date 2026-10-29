"""
The seam between the cloud-neutral orchestration and one cloud's way of getting machines.

Everything in `_task.py` and `_environment.py` (probe every cluster, first wins, cancel the rest,
place, retry on a missed hold) is the same on every cloud. What differs is answered by a
`CapacityProvider`: how to ask a cluster for capacity and know it is there, and how a pod claims it.
`flyteplugins.autoplace.gke` is the first provider; an AKS one would be the same ProvisioningRequest
API with a different class name, an AWS one would need some other mechanism, since Karpenter has no
ask-first API.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from datetime import timedelta
from typing import Awaitable, Callable, Dict, Optional, Protocol, Sequence, Union

from flyte._pod import PodTemplate
from flyte._task_environment import TaskEnvironment

from ._shape import PodSet


@dataclass
class Reservation:
    """
    Outcome of one probe. `ok=False` is a normal result, not an error: a probe never raises for
    "no capacity", so a failed search is never replayed as a failure and the orchestration keeps the
    other clusters' answers. `name`/`namespace` are the provider's handle for the reserved capacity
    (for GKE, the ProvisioningRequest), which `consume` turns into pod metadata.
    """

    queue: str
    ok: bool = False
    name: str = ""
    namespace: str = ""
    reason: str = ""


class CapacityProvider(Protocol):
    """
    What a cloud has to supply for AutoPlace to run on it. Two responsibilities, and only two:

    1. `reserve`: ask one cluster for the pod shapes and report when (and whether) the machines are
       there. It must run as a *child action* in that cluster, typically a small task on one of
       `environments`, so that it is durable across driver retries, cancellable (losing probes are
       cancelled, which must withdraw their request), and runs with the cluster's own credentials.
    2. `consume`: make the placed task's pods land on the reserved machines, given the reservation:
       annotations, node selectors, tolerations merged into the task's own pod template.

    `capacity_hold` is how long the cloud keeps the machines waiting for pods; the orchestration
    bounds the placed task's queue time under it and searches again on a miss. `environments` are
    deployed alongside the task environment. `validate` refuses shapes the cloud cannot provision.
    """

    name: str
    capacity_hold: timedelta

    @property
    def environments(self) -> Sequence[TaskEnvironment]: ...

    def validate(self, podsets: Sequence[PodSet]) -> None: ...

    def reserve(self, queue: str, podsets: Sequence[PodSet], wait: timedelta) -> Awaitable[Reservation]: ...

    def consume(self, pod_template: Optional[PodTemplate], reservation: Reservation) -> PodTemplate: ...


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
