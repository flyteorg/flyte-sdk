"""
Cross-cluster capacity search for flyte ("AutoPlace").

`AutoPlaceEnvironment` is a `flyte.TaskEnvironment` whose tasks ask several clusters for capacity at
call time and run on the first cluster that confirms. The orchestration here is cloud-neutral; how a
cloud is asked, and how the placed pod claims the machines, is a `CapacityProvider`. The provider
shipped today is `flyteplugins.autoplace.gke` (GKE Dynamic Workload Scheduler, flex-start).
"""

from ._environment import AutoPlaceEnvironment
from ._provider import CapacityProvider, Reservation, Selection, register_provider, resolve_provider
from ._race import race
from ._shape import PodSet, podset_from_resources
from ._task import AutoPlaceSettings, AutoPlaceTaskTemplate, NoCapacityError, placement_budget, queue_budget

__all__ = [
    "AutoPlaceEnvironment",
    "AutoPlaceSettings",
    "AutoPlaceTaskTemplate",
    "CapacityProvider",
    "NoCapacityError",
    "PodSet",
    "Reservation",
    "Selection",
    "placement_budget",
    "podset_from_resources",
    "queue_budget",
    "race",
    "register_provider",
    "resolve_provider",
]
