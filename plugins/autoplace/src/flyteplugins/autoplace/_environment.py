from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Sequence, Union

from flyte import TaskEnvironment

from ._provider import CapacityProvider, resolve_provider
from ._shape import PodSet, podset_from_resources
from ._task import BOOKING_MARGIN, AutoPlaceSettings


@dataclass(kw_only=True)
class AutoPlaceEnvironment(TaskEnvironment):
    """
    A task environment whose tasks find their own capacity across clusters before they run.

    Every task call from a parent task sends a capacity probe to each queue in `cluster_queues`. In
    each cluster the probe asks the cloud for this environment's pod shape through the configured
    `provider`; the first cluster to confirm wins, every other queue's request is withdrawn, and the
    task runs there on the reserved machines. The shape comes from `resources`, so nothing about the
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
        cluster_queues=["gke-europe-west4", "gke-us-central1", "gke-us-west1"],
        service_account="autoplace",
        provider=GKEFlexStart(max_run=timedelta(days=2)),   # or just provider="gke"
        search_deadline=timedelta(hours=6),
    )

    @env.task
    async def train(data: str) -> str: ...

    result = await train(data)   # from any task: searches, places, runs
    ```

    Args:
        cluster_queues: Queue names, one per candidate cluster, each routing to exactly that cluster:
            the probe and the placed task are sent to the same queue, so they must land in the same
            cluster.
        service_account: Required. The Kubernetes service account the tasks, the probes and the
            releases run as, in every candidate cluster. The probes create and delete the cloud's
            capacity requests, so it needs the permissions the provider documents (for GKE, a `Role` on
            ProvisioningRequests and PodTemplates).
        provider: How capacity is reserved: a registered provider name (`"gke"`, the default) or a
            `CapacityProvider` instance, which is where provider-specific settings live (for GKE:
            `max_run`, `accelerator`).
        search_deadline: How long the search waits for a cluster to reserve the capacity; it is each
            probe's deadline.

    The placed task's `max_queued_time` is capped at the provider's hold on reserved machines (GKE: 10
    minutes) less two minutes, and its `deadline` at the provider's `max_run`. A task that misses the
    hold fails with `NoCapacityError`.

    Not supported in this version: `queue` (replaced by `cluster_queues`), `plugin_config` (it carries
    the placement settings), a named `pod_template` (there is nothing to merge the placement into),
    `reusable`.
    """

    cluster_queues: Sequence[str]
    provider: Union[str, CapacityProvider] = "gke"
    search_deadline: timedelta = timedelta(hours=1)
    podset: PodSet = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        queues = list(self.cluster_queues)
        if not queues or any(not isinstance(q, str) or not q for q in queues):
            raise ValueError("cluster_queues must be a non-empty list of queue names")
        if len(set(queues)) != len(queues):
            raise ValueError(f"cluster_queues has duplicates: {queues}")
        if not self.service_account:
            raise ValueError(
                "AutoPlaceEnvironment needs `service_account`: the probes run as it and create the cloud's capacity "
                "requests, so it needs the provider's permissions in every candidate cluster"
            )
        if self.queue is not None:
            raise ValueError("AutoPlaceEnvironment picks the queue per call; use `cluster_queues`, not `queue`")
        # A clone carries the settings it was built with; they are rebuilt below from its own fields.
        if self.plugin_config is not None and not isinstance(self.plugin_config, AutoPlaceSettings):
            raise ValueError("AutoPlaceEnvironment sets `plugin_config` itself; it cannot carry another plugin's")
        if isinstance(self.pod_template, str):
            raise ValueError(
                "AutoPlaceEnvironment cannot use a named pod template: the placement has to be merged into the "
                "template, so pass a flyte.PodTemplate"
            )
        if self.reusable is not None:
            raise ValueError("AutoPlaceEnvironment does not support reusable environments yet")
        if self.resources is None:
            raise ValueError("AutoPlaceEnvironment needs `resources`: the capacity request is derived from them")
        if self.search_deadline <= timedelta(0):
            raise ValueError("search_deadline must be positive")

        self.provider = resolve_provider(self.provider)
        if self.provider.capacity_hold <= BOOKING_MARGIN:
            raise ValueError(
                f"provider {self.provider.name!r} holds reserved capacity for {self.provider.capacity_hold}, "
                f"which leaves no time to schedule the task"
            )
        self.cluster_queues = tuple(queues)
        self.podset = podset_from_resources(self.resources, count=1)
        self.provider.validate((self.podset,))
        # Tasks are built as AutoPlaceTaskTemplates through the plugin registry, carrying these settings.
        self.plugin_config = AutoPlaceSettings(
            cluster_queues=self.cluster_queues,
            podsets=(self.podset,),
            search_deadline=self.search_deadline,
            service_account=self.service_account,
            provider=self.provider,
        )
        # The provider's probe runs in the candidate clusters as its own environment(s); depending on them
        # is what gets them deployed alongside this one.
        for env in self.provider.environments:
            self.add_dependency(env)
