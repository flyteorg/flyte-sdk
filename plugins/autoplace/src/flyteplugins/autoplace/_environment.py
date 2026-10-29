from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Optional, Sequence, Union

from flyte._task import _rebuild_as
from flyte._task_environment import TaskEnvironment

from ._provider import CapacityProvider, resolve_provider
from ._shape import PodSet, podset_from_resources
from ._task import AutoPlaceSettings, AutoPlaceTaskTemplate

# Default slack between the cloud's hold on reserved machines and the placed task's queue bound:
# time for the leasor to place the action and the image to pull before the hold expires.
_BOOKING_MARGIN = timedelta(minutes=2)


@dataclass(kw_only=True)
class AutoPlaceEnvironment(TaskEnvironment):
    """
    A task environment whose tasks find their own capacity across clusters before they run.

    Every task call from a parent task fans out a capacity probe to each queue in `clusters`. In
    each cluster the probe asks the cloud for this environment's pod shape through the configured
    `provider`; the first cluster to confirm wins, the other requests are withdrawn, and the task
    runs there on the reserved machines. The shape comes from `resources`, so nothing about the
    request is written by hand, and nothing about the cloud leaks into the task: the backend sees a
    plain task with a queue and a pod template.

    The only provider shipped today is `"gke"` (`flyteplugins.autoplace.gke.GKEFlexStart`: Dynamic
    Workload Scheduler, flex-start with queued provisioning). Adding a cloud means implementing
    `CapacityProvider`; nothing in this class changes.

    Example:
    ```python
    env = AutoPlaceEnvironment(
        name="train",
        resources=flyte.Resources(gpu="H100:8", cpu="32", memory="200Gi"),
        clusters=["eu-west4", "us-central1", "us-west1"],
        provider=GKEFlexStart(max_run=timedelta(days=2)),   # or just provider="gke"
        search_timeout=timedelta(hours=6),
    )

    @env.task
    async def train(data: str) -> str: ...

    result = await train(data)   # from any task: searches, places, runs
    ```

    Args:
        clusters: Union **queue names**, one per candidate cluster (a queue whose `clusters` lists
            exactly that cluster). The SDK knows queues, not clusters, so this is what it routes on.
        provider: How capacity is reserved: a registered provider name (`"gke"`, the default) or a
            `CapacityProvider` instance, which is where provider-specific settings live (for GKE:
            `max_run`, `accelerator`).
        search_timeout: How long each probe waits for its cluster to confirm before giving up on it.
        booking_window: Cap on the placed task's `max_queued_time`. Clouds hold reserved machines
            for a limited time (`provider.capacity_hold`; GKE: 10 minutes) before releasing them, so
            a task not scheduled by then is failed fast and the search runs again, up to
            `max_placements` times. Defaults to the hold minus two minutes.
        max_placements: How many times a placement that misses the booking window is retried with a
            fresh search.

    Requirements on each cluster: whatever the provider documents, plus the Union queue named in
    `clusters`.

    Not supported in this version: `queue` (replaced by `clusters`), a named `pod_template` (there
    is nothing to merge the placement into), `reusable`.
    """

    clusters: Sequence[str]
    provider: Union[str, CapacityProvider] = "gke"
    search_timeout: timedelta = timedelta(hours=1)
    booking_window: Optional[timedelta] = None
    max_placements: int = 2
    podset: PodSet = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        clusters = list(self.clusters)
        if not clusters or any(not isinstance(c, str) or not c for c in clusters):
            raise ValueError("clusters must be a non-empty list of queue names")
        if len(set(clusters)) != len(clusters):
            raise ValueError(f"clusters has duplicates: {clusters}")
        if self.queue is not None:
            raise ValueError("AutoPlaceEnvironment picks the queue per call; use `clusters`, not `queue`")
        if isinstance(self.pod_template, str):
            raise ValueError(
                "AutoPlaceEnvironment cannot use a named pod template: the placement has to be merged into the "
                "template, so pass a flyte.PodTemplate"
            )
        if self.reusable is not None:
            raise ValueError("AutoPlaceEnvironment does not support reusable environments yet")
        if self.resources is None:
            raise ValueError("AutoPlaceEnvironment needs `resources`: the capacity request is derived from them")
        if self.search_timeout <= timedelta(0):
            raise ValueError("search_timeout must be positive")
        if self.max_placements < 1:
            raise ValueError("max_placements must be >= 1")

        self.provider = resolve_provider(self.provider)
        hold = self.provider.capacity_hold
        if self.booking_window is None:
            self.booking_window = max(hold - _BOOKING_MARGIN, hold / 2)
        if not timedelta(0) < self.booking_window <= hold:
            raise ValueError(
                f"booking_window must be within (0, {hold}]: provider {self.provider.name!r} holds reserved "
                f"capacity for {hold}"
            )

        self.clusters = tuple(clusters)
        self.podset = podset_from_resources(self.resources, count=1)
        self.provider.validate((self.podset,))
        # The provider's probe runs in the candidate clusters as its own environment(s); depending on them
        # is what gets them deployed alongside this one.
        for env in self.provider.environments:
            self.add_dependency(env)

    def _settings(self) -> AutoPlaceSettings:
        assert not isinstance(self.provider, str) and self.booking_window is not None
        return AutoPlaceSettings(
            clusters=tuple(self.clusters),
            podsets=(self.podset,),
            search_timeout=self.search_timeout,
            booking_window=self.booking_window,
            provider=self.provider,
            max_placements=self.max_placements,
        )

    def task(self, _func: Optional[Callable] = None, /, **kwargs: Any) -> Any:  # type: ignore[override]
        """Decorate a function as a task; same options as `TaskEnvironment.task`, built as an AutoPlaceTaskTemplate."""

        def rebuild(tmpl: Any) -> AutoPlaceTaskTemplate:
            placed = _rebuild_as(tmpl, AutoPlaceTaskTemplate)
            placed.autoplace = self._settings()  # type: ignore[attr-defined]
            self._add_task(placed)  # replaces the plain template registered by super().task
            return placed  # type: ignore[return-value]

        if _func is None:
            decorator = super().task(**kwargs)
            return lambda f: rebuild(decorator(f))
        return rebuild(super().task(_func, **kwargs))
