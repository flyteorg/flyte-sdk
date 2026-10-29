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

1. For every queue in `clusters`, the provider's probe is submitted as a child task. It runs *in*
   that cluster and asks the cloud for this environment's pod shape (derived from `resources`).
   For GKE that is a `ProvisioningRequest` owned by the probe pod.
2. The first probe to report success wins. The others are cancelled; the provider's request goes
   with them (for GKE, through the ownerReference).
3. The task is submitted to the winning queue with a pod template the provider has marked to
   consume the reserved machines (for GKE: two annotations, the `gke-flex-start` selector, the
   `gke-queued` toleration), with `max_queued_time` capped at `booking_window`. Clouds hold reserved
   machines only briefly (GKE: 10 minutes), so a placement that misses the window fails fast and the
   search runs again.

The backend sees ordinary actions with a queue and a pod template; the task proto is unchanged.
`train.override(queue="us-central1")` pins a cluster and skips the search. Outside a task context
(local runs) the task is a plain function.

## Providers

A provider answers two questions: *how do I reserve capacity in one cluster and know it is there?*
and *how does the placed pod claim it?* It is a small object:

```python
class CapacityProvider(Protocol):
    name: str
    capacity_hold: timedelta  # how long the cloud keeps reserved machines waiting
    environments: Sequence[flyte.TaskEnvironment]  # deployed alongside (the probe's environment)

    def validate(self, podsets) -> None: ...  # refuse shapes the cloud cannot provision
    def reserve(self, queue, podsets, wait) -> Awaitable[Reservation]: ...  # a child task in `queue`'s cluster
    def consume(self, pod_template, reservation) -> flyte.PodTemplate: ...  # pod metadata that claims it
```

`register_provider("name", factory)` makes it available as `provider="name"`. `reserve` must be a
child action (a task) so it is durable across driver retries, cancellable, and uses the cluster's own
credentials. The example `examples/autoplace_smoke.py` has a 20-line provider.

| Cloud | Mechanism | Status |
|---|---|---|
| GKE | ProvisioningRequest, `queued-provisioning.gke.io` (queues until capacity exists, atomic) | `flyteplugins.autoplace.gke.GKEFlexStart` |
| AKS (cluster-autoscaler pools, k8s ≥ 1.33) | same ProvisioningRequest API, `best-effort-atomic-scale-up` (atomic, no queueing) | not written; mostly a different class name |
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
    verbs: ["create", "get", "delete"]
  - apiGroups: [""]
    resources: ["podtemplates"]
    verbs: ["create", "delete"]
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get"]
```

The probe environment is `flyteplugins.autoplace.gke.probe_env`. Its image is read at deploy time, so
`probe_env.image = ...` before deploying works; its service account is fixed into the task when the module
is imported, so use `GKEFlexStart(probe_service_account=...)` to run the probe as another account.

## Not yet

Clustered (JobSet) and Ray environments, reusable environments, TPUs, providers for AKS and AWS.
