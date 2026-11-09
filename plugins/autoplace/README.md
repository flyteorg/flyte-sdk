# flyteplugins-autoplace

Cross-cluster GPU capacity search from the SDK alone: a task asks several clusters for machines at
the same time and runs on the first one that confirms. The orchestration is cloud-neutral; a
**capacity provider** supplies the cloud-specific part. The provider shipped today is GKE's Dynamic
Workload Scheduler (flex-start with queued provisioning).

```python
from datetime import timedelta
import flyte
from flyteplugins.autoplace import AutoPlaceEnvironment
from flyteplugins.autoplace.gke import GKEFlexStart

env = AutoPlaceEnvironment(
    name="train",
    resources=flyte.Resources(gpu="H100:8", cpu="32", memory="200Gi"),
    cluster_queues=["gke-europe-west4", "gke-us-central1", "gke-us-west1"],  # one queue per cluster
    service_account="autoplace",  # needs the provider's RBAC in every cluster (below)
    provider=GKEFlexStart(max_run=timedelta(days=2)),  # or provider="gke" for the defaults
    search_deadline=timedelta(hours=6),
)


@env.task
async def train(data: str) -> str: ...


@driver_env.task
async def main(data: str) -> str:
    return await train(data)  # probes every cluster, runs on the first that reserves capacity
```

## How a call is placed

1. The calling task submits a probe (the provider's `reserve`, a child task) to every queue in
   `cluster_queues`, each with `search_deadline` as its Flyte deadline. Each probe runs *in* its
   cluster, as the environment's `service_account`, and asks the cloud for this environment's pod
   shape (derived from `resources`). For GKE that is a `ProvisioningRequest`, owned by the probe pod
   while it waits. A probe returns once the capacity is there, and fails if the cloud gives up or the
   deadline passes.
2. The first probe to return wins and the others are cancelled. Every queue but the winner's is then
   released by name: each search names its requests up front, so this covers clusters that said no,
   probes that were cancelled, and clusters that provisioned at the same moment as the winner. A
   winning GKE probe hands its request over before exiting, so the request outlives the probe pod.
3. The task is submitted to the winning queue with a pod template the provider has marked to
   consume the reserved machines (for GKE: two annotations, the `gke-flex-start` selector, the
   `gke-queued` toleration). Its `max_queued_time` is capped at the cloud's hold on the machines less
   two minutes (GKE: 8 minutes) and its `deadline` at the provider's `max_run`. A task that misses the
   hold fails with `NoCapacityError`. The winner is released when the task finishes.

The probes and the placed task are child actions with names fixed by the search, so a retried parent
task replays them: it gets the same winner and re-attaches to the task it already submitted, instead
of searching again.

The backend sees ordinary actions with a queue and a pod template; the task proto is unchanged.
`train.override(queue="gke-us-central1")` pins a cluster and skips the search. Outside a task context
(local runs) the task is a plain function. Deleting a cloud request does not stop machines the cloud
is already bringing up for it, so a cluster that loses a close race can boot nodes that stay idle until
its autoscaler removes them. The winner is released by the calling task, after the placed task finishes
or fails; if the calling task dies and is not retried, nothing releases it, and its machines can stay up
until the provider's `max_run` (GKE: `GKEFlexStart(max_run=...)`), so keep `max_run` close to what the
task needs.

## Providers

A provider answers three questions: *how is one cluster asked for capacity?*, *how does the placed
pod claim it?* and *how is it given back?* It is a small object:

```python
class CapacityProvider(Protocol):
    name: str
    capacity_hold: timedelta  # how long the cloud keeps reserved machines waiting for pods
    max_run: timedelta  # how long reserved machines live
    environments: Sequence[flyte.TaskEnvironment]  # deployed alongside (the probe environment)

    def validate(self, podsets) -> None: ...  # refuse shapes the cloud cannot provision
    async def reserve(self, queue, name, podsets, deadline, service_account) -> Reservation: ...
    def consume(self, pod_template, reservation) -> flyte.PodTemplate: ...  # pod metadata that claims it
    def release(self, reservation, service_account) -> Awaitable[None]: ...
```

`register_provider("name", factory)` makes it available as `provider="name"`. `reserve` and `release`
must submit child tasks to the given queue, so they run in that cluster with its own credentials and
are durable across retries of the calling task; `reserve` returns once the capacity is there and raises
otherwise. Releasing something never reserved must succeed. The search across clusters is the same
for every provider.

| Cloud | Mechanism | Status |
|---|---|---|
| GKE | ProvisioningRequest, `queued-provisioning.gke.io` (queues until capacity exists, atomic) | `flyteplugins.autoplace.gke.GKEFlexStart` |
| Azure Kubernetes Service (cluster-autoscaler pools, k8s ≥ 1.33) | same ProvisioningRequest API, `best-effort-atomic-scale-up` (atomic, no queueing) | not written; mostly a different class name |
| AWS (Karpenter) | no ask-first API; Capacity Blocks are an EC2 API | needs a different mechanism |

## GKE per-cluster setup

- A flex-start node pool with queued provisioning for the accelerator
  (`--flex-start --enable-queued-provisioning`, min 0, location policy ANY). Pre-create the pool rather
  than relying on node auto-provisioning, so it exists before a request arrives.
- A queue that routes to exactly that cluster.
- RBAC for the environment's `service_account` in each project namespace (bind this `Role` to it):

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata: { name: autoplace }
rules:
  - apiGroups: ["autoscaling.x-k8s.io"]
    resources: ["provisioningrequests"]
    verbs: ["create", "get", "patch", "delete"]
  - apiGroups: [""]
    resources: ["podtemplates"]
    verbs: ["create", "patch", "delete"]
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get"]
```

The provider deploys one environment, `flyteplugins.autoplace.gke.probe_env` (the probes and the
releases). Its image is read at deploy time, so setting `.image` on it before deploying works.

## Not yet

Clustered (JobSet) and Ray environments, reusable environments, TPUs, providers for Azure and AWS.
