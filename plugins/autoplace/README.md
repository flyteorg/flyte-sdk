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
    clusters=["eu-west4", "us-central1", "us-west1"],  # queue names, one per cluster
    provider=GKEFlexStart(max_run=timedelta(days=2)),  # or provider="gke" for the defaults
    search_timeout=timedelta(hours=6),
)


@env.task
async def train(data: str) -> str: ...


@driver_env.task
async def main(data: str) -> str:
    return await train(data)  # probes every cluster, runs on the first that reserves capacity
```

## How a call is placed

1. A search task (the provider's `select`) submits a probe to every queue in `clusters`. Each probe
   runs *in* its cluster and asks the cloud for this environment's pod shape (derived from
   `resources`). For GKE that is a `ProvisioningRequest`, owned by the probe pod while it waits.
2. The first probe to report success wins and the probes still waiting are cancelled; their requests
   go with them (for GKE, through the ownerReference). A probe that won at the same moment cannot be
   cancelled into giving its machines back, so it is returned as an *extra* and released right away.
   A winning GKE probe hands its request over before exiting, so the request outlives the probe pod.
3. The task is submitted to the winning queue with a pod template the provider has marked to
   consume the reserved machines (for GKE: two annotations, the `gke-flex-start` selector, the
   `gke-queued` toleration), with `max_queued_time` capped at what is left of the cloud's hold
   (GKE: 10 minutes from provisioning) and at most `booking_window`. A placement that misses it fails
   fast and the search runs again. The winner is released when the task finishes.

The search, the hold check and the releases are child actions or traced, so a retried parent task
replays them: it gets the same winner and re-attaches to the task it already submitted, instead of
searching again.

The backend sees ordinary actions with a queue and a pod template; the task proto is unchanged.
`train.override(queue="us-central1")` pins a cluster and skips the search. Outside a task context
(local runs) the task is a plain function.

## Providers

A provider answers three questions: *which cluster got the capacity first?*, *how does the placed pod
claim it?* and *how is it given back?* It is a small object:

```python
class CapacityProvider(Protocol):
    name: str
    capacity_hold: timedelta  # how long the cloud keeps reserved machines waiting
    environments: Sequence[flyte.TaskEnvironment]  # deployed alongside (probe and search environments)

    def validate(self, podsets) -> None: ...  # refuse shapes the cloud cannot provision
    def select(self, queues, podsets, wait) -> Awaitable[Selection]: ...  # a child task; winner + extras
    def consume(self, pod_template, reservation) -> flyte.PodTemplate: ...  # pod metadata that claims it
    def release(self, reservation) -> Awaitable[None]: ...  # a child task in the reservation's cluster
```

`register_provider("name", factory)` makes it available as `provider="name"`. `select` and `release`
must be child actions (tasks), so they are durable across driver retries, and the per-cluster probes
inside `select` must run in their clusters with those clusters' own credentials. `race()` implements
the first-wins logic over a probe per queue for any provider. The example `examples/autoplace_smoke.py`
has a short provider.

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
- RBAC for the `autoplace` service account in each project namespace:

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

The provider deploys two environments: `flyteplugins.autoplace.gke.probe_env` (the probes and the
releases, which need the RBAC above) and `select_env` (the search, which only submits probes and runs as
the run's default account). Their images are read at deploy time, so setting `.image` on them before
deploying works; the probe's service account is fixed into the task when the module is imported, so use
`GKEFlexStart(probe_service_account=...)` to run the probe as another account.

## Not yet

Clustered (JobSet) and Ray environments, reusable environments, TPUs, providers for Azure and AWS.
