# Flyte 2 ↔ SkyPilot Integration Spec

**Status:** Draft for review
**Owners:** Flyte maintainers + Union engineering leadership
**Audience:** Flyte maintainer team (engineering & PM), Union platform engineering, and teams evaluating a Flyte + SkyPilot stack
**Scope:** Two integration horizons — (1) a first-class `flyteplugins-skypilot` connector package, and (2) a longer-horizon architecture in which SkyPilot bootstraps ephemeral Union data planes on demand anywhere, at zero idle cost
**Companion docs:** [SPEC.md](./SPEC.md), [COMPARISON.md](./COMPARISON.md)

---

## TL;DR

This spec covers two integration horizons between Flyte 2 / Union and SkyPilot.

**Horizon 1 — connector plugin (phases 1–7):** A new `flyteplugins-skypilot` package lets users target SkyPilot as a compute backend from within a normal Flyte task or app. The integration follows the same SDK conventions as `flyteplugins.ray`, `flyteplugins.databricks`, and `flyteplugins.vllm`:

- A `SkyPilot` dataclass passed as `plugin_config=` to `flyte.TaskEnvironment` for one-shot tasks.
- A `SkyPilotJob` subclass for managed jobs (spot recovery, multi-region failover).
- A `SkyPilotConnector` that drives the SkyPilot API server via gRPC/REST and reports state back to Flyte.
- A `SkyServeAppEnvironment` for long-running services, mirroring `VLLMAppEnvironment` and `SGLangAppEnvironment`.

**Horizon 2 — ephemeral data planes (phases 8–9):** Rather than running individual tasks on SkyPilot infrastructure, SkyPilot bootstraps an entire Union data plane on demand. SkyPilot provisions cloud VMs, stands up k3s, installs FlytePropeller and the Union operator, and registers the cluster with the Union control plane. When work completes, SkyPilot autostops and the cluster deregisters automatically. The Union control plane is always-on; every data plane is ephemeral and costs nothing when idle.

This second horizon directly addresses the vision raised by Union leadership: _"SkyPilot could launch a Union cluster anywhere"_ — and becomes viable because SkyPilot's API server architecture has matured (see §1.4).

---

## Quickstart

What this looks like to the user. A spot-aware, multi-region GPU job — cron-triggered, retried, with full Flyte lineage — written as one Python file:

```bash
uv pip install flyteplugins-skypilot
```

The SDK integration would look like this:

```python
import flyte
from flyte import Image
from flyte.io import File
from flyteplugins.skypilot import SkyPilotJob

env = flyte.TaskEnvironment(
    name="batch_inference",
    image=Image.from_debian_base(name="scorer").with_pip_packages(
        "torch", "flyteplugins-skypilot",
    ),
    plugin_config=SkyPilotJob(
        accelerators="A100:1",
        use_spot=True,
        regions=["us-west-2", "eu-west-1", "ap-northeast-1"],
        job_recovery="EAGER_NEXT_REGION",
    ),
)


@env.task(
    retries=flyte.RetryStrategy(count=2),
    triggers=flyte.Trigger.cron("0 */6 * * *"),
)
async def batch_score(shard: File) -> File:
    from scorer import run
    return run(shard)
```

No `sky launch`, no YAML, no manual polling, no separate dashboard. The Flyte run UI shows one action; SkyPilot transparently bids on spot A100s across three regions and recovers on preemption. The full surface — one-shot tasks, managed jobs, and SkyServe apps — is in §3.

---

## Audience-specific summaries

### For Flyte maintainers

- **Horizon 1 — no new core concepts.** The connector plugin reuses `AsyncConnector`, `TaskPluginRegistry`, and `AppEnvironment` — the same primitives that back Databricks, Ray, Dask, vLLM, etc. No backend protocol changes required for phases 1–7.
- **Three surfaces, one package.** Following `flyteplugins-databricks` conventions: `SkyPilot`, `SkyPilotJob`, `SkyPilotConnector`, and `SkyServeAppEnvironment` from a single `flyteplugins-skypilot` distribution.
- **Connector deployment.** `flyte-connector-skypilot` runs as its own K8s deployment in the Flyte data plane, talking to a Postgres-backed SkyPilot API server via `SKYPILOT_API_SERVER_ENDPOINT`. Local execution works via `AsyncConnectorExecutorMixin`.
- **IDL surface.** Surface layer piggybacks on the connector custom-config dict. Bridge layer introduces `flyteidl2.plugins.skypilot` proto so launch specs are typed end-to-end. See §8.
- **Horizon 2 — two new primitives needed.** A runtime cluster registration API for FlyteAdmin (or a Union operator webhook), and a pre-baked VM image that reduces bootstrap latency to 2–5 minutes. See §12 for the full design.
- **Risk profile.** Bounded in both horizons. We do not fork SkyPilot or extend Propeller. In Horizon 1, the connector is the only failure boundary. In Horizon 2, an ephemeral data plane failure affects only the runs scheduled on it — the control plane and other data planes are unaffected.

### For Flyte users

- **One pipeline, many clouds.** Use Flyte for durable pipelines, typed data, lineage, retries, caching, and triggers. Use SkyPilot for finding cheap or available GPU capacity across clouds and clusters. Your team writes one Python task; the platform handles the rest.
- **No data-plane re-architecture (Horizon 1).** SkyPilot tasks show up in the same Flyte run UI with the same logs, lineage, and replay story as any other task. Visibility is not lost when work runs off-cluster.
- **Zero idle cost (Horizon 2).** Data planes spin up for a run and disappear when idle. There are no standing GPU clusters to maintain or pay for between factory runs. Every run still has a full lineage record in the Union control plane.
- **Spot-aware, multi-region, managed.** SkyPilot's spot-recovery and managed-job features are exposed natively in both horizons: a `Trigger.cron(...)` pipeline drives a `SkyPilotJob(use_spot=True, regions=[…])` and appears as a single deterministic action in Flyte history.
- **Deployment options.** Self-host Flyte + SkyPilot API server, or use Union-hosted control plane + bring-your-own SkyPilot API server. In Horizon 2, Union can optionally host the SkyPilot API server as a managed add-on.

---

## 1. Background and motivation

### 1.1 What SkyPilot does well

[SkyPilot](https://docs.skypilot.co/en/latest/docs/index.html) is a compute control plane for AI workloads. From an authoring perspective, a user writes a `sky.Task` (or YAML) describing `resources`, `setup`, and `run`, and SkyPilot:

- Provisions VMs or Kubernetes pods on any of 20+ supported clouds plus Kubernetes/Slurm.
- Bin-packs and auto-fails-over across regions, zones, and accelerator types until it finds capacity.
- Manages spot/preemptible recovery, autostop, and idle teardown.
- Exposes a centralized [API server](https://docs.skypilot.co/en/latest/reference/api-server/api-server.html) so teams can share clusters and a single dashboard.
- Provides `sky serve` for autoscaled HTTP services with replica management across clouds.

SkyPilot is *not* an orchestrator: it does not own DAGs, lineage, typed I/O, semantic caching, durable execution, triggers, or audit trails.

### 1.2 Where Flyte 2 has gaps

In the FLYRES context (see [SPEC.md](./SPEC.md)), several requirements push beyond what stock Flyte 2 covers today:

- **Multi-cloud GPU scheduling.** Flyte schedules within a Kubernetes cluster (or cluster pool). It does not directly bid for spot capacity across AWS, GCP, and a neocloud and pick whichever is cheapest right now.
- **Multi-cluster Kubernetes.** Flyte cluster pools exist, but they are coarse-grained. SkyPilot has a richer "try this cluster, fall back to that one" failover story.
- **SSH-style interactive iteration.** Flyte's devbox helps here, but researchers who already work in `sky launch ... && ssh mycluster` will not switch unless they can keep that ergonomic.

The proposed integration closes these gaps by delegating to SkyPilot for the launching and lifecycle of compute, while Flyte continues to own the pipeline, lineage, and durability layer.

### 1.3 Tool boundaries

```mermaid
flowchart LR
    classDef flyte fill:#e6f3ff,stroke:#0066cc,stroke-width:2px,color:#000
    classDef bridge fill:#fff4e6,stroke:#cc7a00,stroke-width:2px,color:#000
    classDef sky fill:#f0e6ff,stroke:#6633cc,stroke-width:2px,color:#000
    classDef infra fill:#e6ffe6,stroke:#009933,stroke-width:2px,color:#000

    subgraph Flyte["**Flyte 2**"]
        direction LR
        F1["DAG / runs / actions"]:::flyte
        F2["Typed I/O &amp; lineage"]:::flyte
        F3["Semantic cache"]:::flyte
        F4["Triggers (cron, event)"]:::flyte
        F5["Retries / replay"]:::flyte
        F6["Audit trail"]:::flyte
    end

    Bridge["flyteplugins-skypilot"]:::bridge

    subgraph Sky["**SkyPilot**"]
        direction LR
        S1["Multi-cloud provisioning"]:::sky
        S2["Spot recovery"]:::sky
        S3["Managed jobs"]:::sky
        S4["Kubernetes / Slurm / VMs"]:::sky
        S5["sky serve"]:::sky
        S6["API server"]:::sky
    end

    subgraph Infra["**Execution targets**"]
        direction LR
        I1["AWS"]:::infra
        I2["GCP"]:::infra
        I3["Azure"]:::infra
        I4["Kubernetes"]:::infra
        I5["Slurm"]:::infra
        I6["Neoclouds"]:::infra
    end

    Flyte --> Bridge --> Sky --> Infra
```

Flyte owns the *what* (pipeline graph, data, lineage). SkyPilot owns the *where* (which cloud, which region, which accelerator). The plugin is the contract between them.

### 1.4 SkyPilot architecture: recent developments

SkyPilot previously stored all state in local files (`~/.sky/`) on the machine running
`sky`, which made it impractical to share a single SkyPilot instance across a team or
to recover from API server failures. That constraint has been fully resolved:

- **Postgres-backed HA API server.** SkyPilot's API server now supports an external
  PostgreSQL database (`db: postgresql://…` in `~/.sky/config.yaml` or via Helm's
  `apiService.dbConnectionString`). State is decoupled from the pod, so rolling
  upgrades and pod rescheduling do not lose cluster or job metadata.
- **Zero-downtime rolling upgrades.** With Postgres backing and `upgradeStrategy:
  RollingUpdate`, the API server can be upgraded without dropping in-flight job
  tracking. The old pod is terminated only after the new one is healthy.
- **Cluster bootstrap from raw VMs.** `sky local up --ips <ips.txt> --ssh-user …
  --ssh-key-path …` deploys a lightweight k3s Kubernetes cluster on any list of
  SSH-accessible machines, installs the NVIDIA GPU Operator if GPUs are present,
  exposes the Kubernetes API server on port 6443, and configures local `kubectl` — the
  entire cluster bootstrap in a single command.

These three developments are what make Horizon 2 (ephemeral Union data planes) viable.
They are documented at:
- [API server HA](https://docs.skypilot.co/en/latest/reference/api-server/api-server-upgrade.html)
- [Existing machines / k3s bootstrap](https://docs.skypilot.co/en/latest/reservations/existing-machines.html)
- [State management](https://docs.skypilot.co/en/pool-docs/reference/architecture/state.html)

---

## 2. Architecture

### 2.1 Runtime topology

```mermaid
flowchart LR
    subgraph User["Authoring"]
        Code["Python task w/<br/>plugin_config=SkyPilot(...)"]
    end

    subgraph FlytePlane["Flyte control + data plane"]
        Propeller["Propeller<br/>(workflow exec)"]
        ConnSvc["flyte-connector-skypilot<br/>(AsyncConnector deployment)"]
        UI["Flyte UI<br/>(logs, lineage, replays)"]
    end

    subgraph Sky["SkyPilot control plane"]
        SkyAPI["SkyPilot API server<br/>(remote, multi-tenant)"]
    end

    subgraph Clouds["Execution targets"]
        K8s["Kubernetes<br/>(BYOC clusters)"]
        AWS["AWS / GCP / Azure"]
        Neo["Neoclouds, Slurm, VMs"]
    end

    Code -->|flyte deploy| Propeller
    Propeller -->|task_type=skypilot| ConnSvc
    ConnSvc -->|create / get / delete<br/>via skypilot SDK| SkyAPI
    SkyAPI -->|provisions| K8s
    SkyAPI -->|provisions| AWS
    SkyAPI -->|provisions| Neo
    K8s -.->|stdout, status| SkyAPI
    SkyAPI -.->|status, logs| ConnSvc
    ConnSvc -.->|Resource phase,<br/>log links| Propeller
    Propeller --> UI
```

Key properties:

- **Single failure boundary.** Only `flyte-connector-skypilot` is on the path between Flyte and SkyPilot. The propeller never blocks on SkyPilot directly.
- **No data-plane changes.** The Flyte data plane sees these tasks as opaque async connector tasks. Logs and dashboard links are surfaced through `TaskLog` entries, identical to how Databricks links the Databricks console.
- **Storage stays in Flyte.** Inputs are materialized via Flyte's `RawDataPath`; outputs are written to the same object store. SkyPilot's `file_mounts` are an optional optimization, not the data path.

### 2.2 SDK module layout

The package follows the standard `flyteplugins-<name>` convention:

```
plugins/skypilot/
├── README.md
├── pyproject.toml             # name = "flyteplugins-skypilot"
├── src/
│   └── flyteplugins/
│       └── skypilot/
│           ├── __init__.py    # public exports
│           ├── task.py        # SkyPilot / SkyPilotJob config + TaskTemplate
│           ├── connector.py   # SkyPilotConnector(AsyncConnector)
│           ├── app.py         # SkyServeAppEnvironment
│           ├── _models.py     # internal dataclasses (Resources, FileMount, etc.)
│           └── _idl.py        # proto wrappers (added in the bridge layer)
└── tests/
```

This is byte-for-byte the same layout as `plugins/databricks/`, `plugins/ray/`, and `plugins/vllm/`.

---

## 3. SDK API design

Every surface below is designed to mirror an existing plugin convention so that users and maintainers do not need a new mental model.

### 3.1 `SkyPilot` — task plugin config (Ray/Dask convention)

Pattern: a `@dataclass` config attached to a `TaskEnvironment` via `plugin_config=`. This is the same shape as `Dask`, `RayJobConfig`, `Spark`, and `Databricks`.

```python
# flyteplugins/skypilot/task.py (excerpt)
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import flyte
from flyte import Resources, Secret


@dataclass
class FileMount:
    """A SkyPilot file_mount entry. Mirrors the YAML schema."""
    target: str
    source: Optional[str] = None        # local path, s3://, gs://
    mode: Optional[str] = None          # "MOUNT", "COPY", "MOUNT_CACHED"


@dataclass
class SkyPilot:
    """
    Configuration for running a Flyte task on infrastructure provisioned by SkyPilot.

    Tasks in an environment with this plugin_config are launched as SkyPilot
    tasks via the configured SkyPilot API server. SkyPilot handles provisioning,
    setup, and teardown; Flyte handles inputs, outputs, lineage, and retries.

    Attributes:
        accelerators: e.g. "H100:8" or ["H100:8", "A100:8"] for failover.
        infra: SkyPilot infra spec, e.g. "k8s", "aws/us-east-1", "k8s/coreweave".
        cpus: CPU request, e.g. "4+".
        memory: Memory in GB, e.g. "32+".
        use_spot: Whether to bid for spot/preemptible capacity.
        regions: Optional ordered list of regions for failover.
        max_hourly_cost: Cap on hourly price (USD) when selecting infra.
        cluster_name: Reuse an existing SkyPilot cluster by name.
        autostop: Idle minutes before auto-stop; True for default (5 min).
        setup: Shell setup commands run once per cluster.
        run: Shell run commands. If None, the Flyte task entrypoint is used.
        file_mounts: List of FileMount entries.
        envs: Environment variables forwarded to the task.
        secrets: Flyte secrets injected into the SkyPilot task environment.
        api_server_endpoint: Overrides the connector default.
    """
    accelerators: Optional[Union[str, List[str]]] = None
    infra: Optional[str] = None
    cpus: Optional[str] = None
    memory: Optional[str] = None
    use_spot: bool = False
    regions: Optional[List[str]] = None
    max_hourly_cost: Optional[float] = None
    cluster_name: Optional[str] = None
    autostop: Union[bool, int, str, None] = True
    setup: Optional[str] = None
    run: Optional[str] = None
    file_mounts: List[FileMount] = field(default_factory=list)
    envs: Optional[Dict[str, str]] = None
    secrets: Optional[List[Secret]] = None
    api_server_endpoint: Optional[str] = None
```

**Why this shape.** It is identical in spirit to `RayJobConfig` and `Databricks`: a flat dataclass that maps directly to the upstream tool's first-class concepts (in this case the [SkyPilot YAML spec](https://docs.skypilot.co/en/latest/reference/yaml-spec.html)).

**Registration.** Following Ray/Dask:

```python
# flyteplugins/skypilot/task.py (continued)
from flyte.connectors import AsyncConnectorExecutorMixin
from flyte.extend import AsyncFunctionTaskTemplate, TaskPluginRegistry


@dataclass(kw_only=True)
class SkyPilotTask(AsyncConnectorExecutorMixin, AsyncFunctionTaskTemplate):
    plugin_config: SkyPilot
    task_type: str = "skypilot"

    def custom_config(self, sctx):
        # Serialize SkyPilot dataclass -> dict; the connector reconstitutes it
        # into a sky.Task at create() time. See _models.py for the round-trip.
        return _serialize_skypilot(self.plugin_config, sctx)


TaskPluginRegistry.register(SkyPilot, SkyPilotTask)
```

`AsyncConnectorExecutorMixin` is the same mixin used by Databricks; it provides the local-execution simulation so users can run a SkyPilot task locally during development.

#### Author-side usage

```python
import flyte
from flyte import Image, Secret
from flyteplugins.skypilot import SkyPilot

image = (
    Image.from_debian_base(name="trainer")
    .with_pip_packages("torch", "transformers", "flyteplugins-skypilot")
)

env = flyte.TaskEnvironment(
    name="train_env",
    image=image,
    plugin_config=SkyPilot(
        accelerators="H100:8",
        infra="k8s/coreweave",
        use_spot=True,
        regions=["us-west-2", "eu-west-1"],
        autostop=10,
        secrets=[Secret(key="HF_TOKEN", group="hf")],
    ),
)


@env.task
async def finetune(dataset: flyte.io.Dir, epochs: int = 3) -> flyte.io.Dir:
    # Body runs *inside* the SkyPilot-provisioned environment. SkyPilot's
    # `setup`/`run` are inferred from the Flyte task entrypoint; users only
    # set them explicitly when they need shell-level customization.
    from train import train_model
    return train_model(dataset, epochs=epochs)
```

That is the entire authoring surface for a one-shot SkyPilot task. There is no `sky launch`, no YAML, no manual polling.

### 3.2 `SkyPilotJob` — managed jobs subclass (Databricks-of-Spark convention)

Just like `Databricks(Spark)` extends `Spark` with cluster-specific fields, `SkyPilotJob(SkyPilot)` extends `SkyPilot` with the additional fields that the [managed jobs API](https://docs.skypilot.co/en/latest/reference/api.html) takes. This communicates to users: "you're doing the same thing, but with auto-recovery."

```python
@dataclass
class SkyPilotJob(SkyPilot):
    """
    Configuration for a SkyPilot *managed job*.

    Managed jobs are auto-restarted on spot preemption and can fail over
    across infra. Use this for long-running batch work where the cost
    of restart is bounded.

    Additional attributes:
        job_recovery: SkyPilot recovery strategy, e.g. "EAGER_NEXT_REGION".
        max_restarts_on_errors: How many times to retry on user errors.
        priority: Job priority (0..1000) within the API server queue.
        detach_run: Whether to return immediately after submission.
    """
    job_recovery: str = "EAGER_NEXT_REGION"
    max_restarts_on_errors: int = 0
    priority: int = 100
    detach_run: bool = True


TaskPluginRegistry.register(SkyPilotJob, SkyPilotJobTask)  # subclass routing
```

Authoring:

```python
env = flyte.TaskEnvironment(
    name="batch_inference",
    image=image,
    plugin_config=SkyPilotJob(
        accelerators="A100:1",
        use_spot=True,
        regions=["us-west-2", "eu-west-1", "ap-northeast-1"],
        job_recovery="EAGER_NEXT_REGION",
        max_restarts_on_errors=3,
    ),
)


@env.task(
    retries=flyte.RetryStrategy(count=2),
    triggers=flyte.Trigger.cron("0 */6 * * *"),
)
async def batch_score(shard: flyte.io.File) -> flyte.io.File:
    ...
```

### 3.3 `SkyPilotConnector` — the async connector (Databricks/BigQuery convention)

This is the heart of the integration. It implements `AsyncConnector` (the same interface used by `DatabricksConnector` and `BigQueryConnector`) and talks to SkyPilot via its Python SDK.

```python
# flyteplugins/skypilot/connector.py (sketch)
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import sky
from flyte import logger
from flyte.connectors import (
    AsyncConnector,
    ConnectorRegistry,
    Resource,
    ResourceMeta,
)
from flyte.connectors.utils import convert_to_flyte_phase
from flyteidl2.core.execution_pb2 import TaskExecution, TaskLog
from flyteidl2.core.tasks_pb2 import TaskTemplate


@dataclass
class SkyPilotJobMetadata(ResourceMeta):
    api_server: str
    cluster_name: Optional[str]   # for one-shot tasks
    job_id: int                   # SkyPilot job id
    managed: bool                 # True if launched via sky.jobs.launch


class SkyPilotConnector(AsyncConnector):
    name = "SkyPilot Connector"
    task_type_name = "skypilot"     # also handles "skypilot_job" via _task_type_versions
    metadata_type = SkyPilotJobMetadata

    async def create(
        self,
        task_template: TaskTemplate,
        inputs: Optional[Dict[str, Any]] = None,
        skypilot_token: Optional[str] = None,
        **kwargs,
    ) -> SkyPilotJobMetadata:
        spec = _deserialize_skypilot(task_template.custom)
        sky_task = _build_sky_task(spec, task_template, inputs)
        api_server = spec.api_server_endpoint or os.environ["SKYPILOT_API_SERVER_ENDPOINT"]
        sky.api.login(endpoint=api_server, token=skypilot_token)

        if task_template.type == "skypilot_job":
            request_id = sky.jobs.launch(sky_task, name=task_template.id.name)
            job_id, _ = sky.stream_and_get(request_id)
            return SkyPilotJobMetadata(api_server=api_server, cluster_name=None,
                                        job_id=job_id, managed=True)
        else:
            request_id = sky.launch(
                task=sky_task,
                cluster_name=spec.cluster_name or _autoname(task_template),
                detach_run=True,
            )
            job_id, handle = sky.stream_and_get(request_id)
            return SkyPilotJobMetadata(
                api_server=api_server,
                cluster_name=handle.get_cluster_name(),
                job_id=job_id,
                managed=False,
            )

    async def get(self, resource_meta: SkyPilotJobMetadata, **kwargs) -> Resource:
        if resource_meta.managed:
            status = sky.jobs.queue(job_ids=[resource_meta.job_id])[0]
        else:
            status = sky.job_status(resource_meta.cluster_name, [resource_meta.job_id])[
                resource_meta.job_id
            ]
        phase = convert_to_flyte_phase(_skypilot_status_to_string(status))
        log_link = TaskLog(
            uri=_dashboard_url(resource_meta),
            name="SkyPilot Dashboard",
            ready=True,
            link_type=TaskLog.DASHBOARD,
        )
        return Resource(phase=phase, message=str(status), log_links=[log_link])

    async def delete(self, resource_meta: SkyPilotJobMetadata, **kwargs):
        if resource_meta.managed:
            sky.jobs.cancel(job_ids=[resource_meta.job_id])
        elif resource_meta.cluster_name:
            sky.cancel(resource_meta.cluster_name, job_ids=[resource_meta.job_id])

    async def get_logs(self, resource_meta, token: str = "", **kwargs):
        # Stream SkyPilot logs page-by-page using sky.tail_logs(follow=False)
        async for page in _paginate_logs(resource_meta, token):
            yield page


ConnectorRegistry.register(SkyPilotConnector())
```

The shape — `create / get / delete / get_logs` — is exactly the contract Flyte 2 already defines for [connectors](https://www.union.ai/docs/v2/flyte/integrations/) and is the same one Databricks uses today.

### 3.4 `SkyServeAppEnvironment` — long-running services (vLLM/SGLang convention)

For services (model endpoints, web apps), follow the convention set by `VLLMAppEnvironment` and `SGLangAppEnvironment`: a subclass of `flyte.app.AppEnvironment` whose `__post_init__` wires the right `setup`/`run` commands and forwards to `sky serve up` rather than running in-cluster.

```python
# flyteplugins/skypilot/app.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import flyte.app
from flyte import Image, Resources, Secret


@dataclass(kw_only=True)
class SkyServeAppEnvironment(flyte.app.AppEnvironment):
    """
    Deploy a Flyte app as a SkyPilot service (`sky serve`).

    SkyServe handles cross-cloud replica placement, autoscaling, and load
    balancing; Flyte handles app definition, secrets, and the URL surface
    visible in the Flyte UI.

    Attributes:
        replicas: Replica count or autoscaler config (min, max, qps target).
        accelerators: Per-replica accelerator spec (e.g. "H100:1").
        infra: SkyPilot infra string (multi-cloud-aware).
        readiness_probe: Path or full probe config.
    """
    replicas: int | "Autoscaling" = 2
    accelerators: Optional[str] = None
    infra: Optional[str] = None
    readiness_probe: str = "/health"
    sky_setup: Optional[str] = None
    sky_run: Optional[str] = None
```

Authoring (lifted directly from the vLLM convention):

```python
from flyteplugins.skypilot import SkyServeAppEnvironment

app_env = SkyServeAppEnvironment(
    name="customer-llm",
    image=image,
    accelerators="H100:1",
    infra="k8s/coreweave",
    replicas=flyte.Autoscaling(min=2, max=8, target_qps=20),
    secrets=[Secret(key="HF_TOKEN", group="hf")],
    sky_run="vllm serve meta-llama/Meta-Llama-3.1-8B --port 8000",
    port=8000,
)
```

`flyte deploy` registers the app; the connector calls `sky.serve.up(...)` and pins the resulting URL into the Flyte app's `Link`. From the operator's standpoint, this is the same flow as deploying a vLLM app today.

### 3.5 Public exports

```python
# flyteplugins/skypilot/__init__.py
from flyteplugins.skypilot.app import SkyServeAppEnvironment
from flyteplugins.skypilot.connector import SkyPilotConnector
from flyteplugins.skypilot.task import FileMount, SkyPilot, SkyPilotJob

__all__ = [
    "FileMount",
    "SkyPilot",
    "SkyPilotJob",
    "SkyPilotConnector",
    "SkyServeAppEnvironment",
]
```

---

## 4. Capability mapping

A direct cross-reference. Anything in the left column maps to the right column 1:1; new authoring surface is intentionally minimal.

| Flyte 2 concept                          | SkyPilot concept                                                       | How the plugin handles it                                          |
|------------------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------|
| `TaskEnvironment(plugin_config=SkyPilot)`| `sky.Task(resources=...)`                                              | Connector translates dataclass → `sky.Task` in `create()`          |
| `Resources(gpu="H100:8")`                | `accelerators="H100:8"`                                                | Auto-derived from `plugin_config.accelerators`                     |
| `flyte.io.File` / `flyte.io.Dir` input   | Object-store URI                                                       | Materialized via Flyte; no SkyPilot `file_mount` needed by default |
| `Secret(key=..., group=...)`             | `secrets:` block / env vars                                            | Connector injects via SkyPilot `envs`                              |
| `Trigger.cron(...)`                      | (n/a)                                                                  | Flyte schedules; each fire creates a fresh SkyPilot task           |
| `RetryStrategy`                          | `max_restarts_on_errors`                                               | Flyte handles outer retry; managed job handles inner               |
| `cache="auto"`                           | (n/a)                                                                  | Flyte's semantic cache short-circuits before SkyPilot is called    |
| `Action.logs` / Flyte UI                 | `sky dashboard`, `sky logs`                                            | `get_logs()` streams SkyPilot logs into Flyte UI                   |
| `flyte.deploy()` on an `AppEnvironment`  | `sky serve up`                                                         | `SkyServeAppEnvironment` connector calls `sky.serve.up`            |
| Devbox (`flyte.with_runcontext`)         | `ssh mycluster` / SkyPilot pods                                        | Control-plane layer: Flyte devbox can opt into a SkyPilot-managed pod |

---

## 5. Execution lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Flyte as Flyte Propeller
    participant Conn as flyte-connector-skypilot
    participant Sky as SkyPilot API server
    participant Cloud as Target Cloud / K8s
    participant Store as Object Store

    User->>Flyte: flyte run finetune.py main
    Flyte->>Flyte: Materialize inputs to object store
    Flyte->>Conn: AsyncConnector.create(task_template, inputs)
    Conn->>Sky: sky.launch(sky_task)   (gRPC over API server)
    Sky->>Cloud: Provision GPU pod / VM
    Cloud-->>Sky: Cluster up, setup done
    Sky-->>Conn: request_id -> job_id, cluster_name
    Conn-->>Flyte: ResourceMeta(job_id, cluster_name)
    loop poll
        Flyte->>Conn: AsyncConnector.get(meta)
        Conn->>Sky: sky.job_status(...)
        Sky-->>Conn: RUNNING / SUCCEEDED / FAILED
        Conn-->>Flyte: Resource(phase, log_links)
    end
    Cloud->>Store: Write outputs (model weights, metrics)
    Sky-->>Conn: SUCCEEDED, outputs path
    Conn-->>Flyte: Resource(phase=SUCCEEDED, outputs)
    Flyte->>User: Action SUCCEEDED, lineage stored
    Note over Sky,Cloud: autostop / autodown tears down cluster
```

The shape is identical to the existing Databricks lifecycle; only the verbs (`sky.launch` vs. `runs/submit`) differ.

---

## 6. Worked end-to-end example

Below is a full pipeline using the proposed API. It mirrors the structure of `examples/reference_stacks/flyres_stack/08_end_to_end_ml_pipeline.py` so a maintainer can do a side-by-side review against current conventions.

```python
import flyte
from flyte import Image, Resources, Secret
from flyte.io import Dir, File
from flyteplugins.skypilot import SkyPilot, SkyPilotJob, SkyServeAppEnvironment


image = (
    Image.from_debian_base(name="flyres", python_version=(3, 12))
    .with_pip_packages("torch", "transformers", "datasets", "flyteplugins-skypilot")
)

prep_env = flyte.TaskEnvironment(
    name="prep_env",
    image=image,
    resources=Resources(cpu="8", memory="32Gi"),
)

train_env = flyte.TaskEnvironment(
    name="train_env",
    image=image,
    plugin_config=SkyPilot(
        accelerators="H100:8",
        infra="k8s/coreweave",
        use_spot=True,
        regions=["us-west-2", "eu-west-1"],
        autostop=15,
        secrets=[Secret(key="HF_TOKEN", group="hf")],
    ),
)

eval_env = flyte.TaskEnvironment(
    name="eval_env",
    image=image,
    plugin_config=SkyPilotJob(
        accelerators="A100:1",
        use_spot=True,
        regions=["us-west-2", "eu-west-1", "ap-northeast-1"],
        job_recovery="EAGER_NEXT_REGION",
        max_restarts_on_errors=3,
    ),
)

serve_env = SkyServeAppEnvironment(
    name="customer-llm",
    image=image,
    accelerators="H100:1",
    infra="k8s/coreweave",
    replicas=flyte.Autoscaling(min=2, max=8, target_qps=25),
    secrets=[Secret(key="HF_TOKEN", group="hf")],
    sky_run="vllm serve $MODEL_PATH --port 8000",
    port=8000,
)


@prep_env.task(cache="auto")
async def prepare(raw: Dir) -> Dir:
    ...


@train_env.task(retries=flyte.RetryStrategy(count=2))
async def finetune(dataset: Dir, epochs: int) -> Dir:
    ...


@eval_env.task
async def evaluate(weights: Dir, eval_set: Dir) -> File:
    ...


@prep_env.task(
    triggers=flyte.Trigger.cron("0 2 * * *", timezone="UTC"),
)
async def main(raw: Dir, eval_set: Dir) -> File:
    dataset = await prepare(raw)
    weights = await finetune(dataset, epochs=3)
    report = await evaluate(weights, eval_set)
    return report


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(main, serve_env)
```

What this buys the user, end-to-end:

- `prepare` runs in the local cluster pool (no SkyPilot).
- `finetune` is launched on SkyPilot, on spot H100s, across two regions, with code synced from the Flyte code bundle.
- `evaluate` is a SkyPilot *managed job*: if it gets preempted, SkyPilot eagerly retries in the next region without Flyte intervention.
- `serve_env` is deployed via `sky serve` and surfaces a single auth-protected URL in the Flyte app dashboard.
- The whole pipeline is one cron-triggered Flyte run, with one lineage graph and one audit trail.

---

## 7. Implementation details for maintainers

Each subsection below maps to a phase in §8. The intent is to keep the implementation per phase narrow enough that any phase can be reviewed, merged, and shipped as an independent unit.

### 7.1 Surface layer (phases 1–2)

- **Phase 1 — subprocess pattern.** Documentation and reference examples only. No new code in `plugins/`. Users wrap `sky launch` from a normal Flyte task using `flyte.extras.ContainerTask` or a shell-out. This exists to validate demand and surface authoring frictions.
- **Phase 2 — in-process SDK plugin.** `flyteplugins-skypilot` ships with the `SkyPilot` / `SkyPilotJob` dataclasses and a `SkyPilotTask` that calls the SkyPilot Python SDK in-process from the user's task pod. Reuses the existing connector custom-config dict; no IDL changes. `AsyncConnectorExecutorMixin` keeps `flyte run --local` working. No new deployments — the user's container ships `skypilot` and the API server endpoint comes from a task env var or secret.

### 7.2 Bridge layer (phases 3–4)

- **Phase 3 — async connector deployment.** Promote SkyPilot to a true async connector. One new K8s deployment (`flyte-connector-skypilot`) added to the Helm chart's `connectors` map. Centralizes credentials, isolates failure boundaries, and removes the dependency on every user image carrying `skypilot`.
- **Phase 4 — typed IDL.** Add `flyteidl2/plugins/skypilot.proto` with `SkyPilotJob` and `SkyPilotServe` messages so launch specs are typed end-to-end. Generate Python stubs in `flyteplugins-skypilot`; switch `custom_config` from `MessageToDict(dict)` to `MessageToDict(SkyPilotJob(...))` — same pattern as `flyteidl2.plugins.spark_pb2` used by Spark and Databricks.

### 7.3 Control-plane layer (phases 5–7)

- **Phase 5 — `SkyServeAppEnvironment`.** Integrate `sky serve` with the Flyte app/serving lifecycle. Replica/autoscaling settings translate to SkyServe's replica config. `flyte deploy` registers the app, the connector calls `sky.serve.up(...)`, and the resulting URL is pinned into the Flyte app's `Link`.
- **Phase 6 — devbox SkyPilot backend.** `flyte devbox launch --plugin skypilot --accelerators H100:8` ssh-tunnels into a SkyPilot-provisioned pod. The Flyte CLI uses the connector to call `sky.launch(... cluster_name=devbox-<user>-<id>)`, then opens an SSH tunnel. This closes the "honest tension" called out in [SPEC.md](./SPEC.md) without rebuilding SkyPilot's strengths.
- **Phase 7 — lineage, cost, and telemetry.** Surface SkyPilot cost telemetry, spot-recovery events, and multi-cluster routing decisions in the Flyte UI and run graph. Optional control-plane affinity (mapping Flyte queues to SkyPilot priorities). Touches backend and UI; intentionally last.

### 7.4 Dependencies

```toml
# plugins/skypilot/pyproject.toml
[project]
name = "flyteplugins-skypilot"
dynamic = ["version"]
description = "SkyPilot plugin for Flyte 2"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "flyte",
    "skypilot>=0.7.0",
]

[project.optional-dependencies]
aws    = ["skypilot[aws]>=0.7.0"]
gcp    = ["skypilot[gcp]>=0.7.0"]
azure  = ["skypilot[azure]>=0.7.0"]
k8s    = ["skypilot[kubernetes]>=0.7.0"]
all    = ["skypilot[all]>=0.7.0"]
```

### 7.5 Auth and secrets

- The connector reads two values at startup:
  - `SKYPILOT_API_SERVER_ENDPOINT` (env var, set on the connector deployment).
  - `skypilot_token` (Flyte secret, injected at connect time via `Secret(key="skypilot_token")`).
- Per-task overrides live on the `SkyPilot` dataclass (`api_server_endpoint`) — useful for customers running per-team SkyPilot API servers.
- Cloud credentials (AWS, GCP keys, kube configs) are owned by SkyPilot itself, not Flyte. Flyte never sees them.

### 7.6 Multi-tenancy

- Each Flyte project maps to a SkyPilot API server user via a configurable resolver in the connector. Default: one shared service account; opt-in: per-project tokens.
- Cluster naming is namespaced: `flyte-<project>-<domain>-<action_id>` to avoid collisions on a shared SkyPilot dashboard.

### 7.7 Failure modes & guardrails

| Failure                                  | Behavior                                                              |
|------------------------------------------|-----------------------------------------------------------------------|
| SkyPilot API server unreachable          | Connector returns `RUNNING` with backoff; surfaces `SkyDownError`     |
| Connector pod crash                      | K8s restarts; `get()` is idempotent so polling resumes from job_id    |
| Spot preemption (one-shot task)          | Connector reports `FAILED`; Flyte retry policy re-launches            |
| Spot preemption (managed job)            | SkyPilot recovers transparently; Flyte sees a single `RUNNING` window |
| Cloud quota exhaustion                   | SkyPilot fails over to next region/cloud; recorded in log links       |
| Long autostop after success              | Connector flips `phase=SUCCEEDED` immediately; teardown is async      |

---

## 8. Build phases

The work is organized as a progression from **surface-level integration** (SDK only, no new infra) to **deep control-plane integration** (devbox, UI, backend telemetry). Each phase is independently shippable, each de-risks the next, and each later phase can be deferred or skipped if the value proven by earlier phases does not warrant the investment.

### 8.1 Sequencing

```mermaid
flowchart LR
    classDef surface fill:#e6f3ff,stroke:#0066cc,stroke-width:2px,color:#000
    classDef bridge fill:#fff4e6,stroke:#cc7a00,stroke-width:2px,color:#000
    classDef deep fill:#ffe6e6,stroke:#cc0000,stroke-width:2px,color:#000
    classDef h2 fill:#f0e6ff,stroke:#6600cc,stroke-width:2px,color:#000

    P1["**Phase 1**<br/>Subprocess pattern<br/>(docs + reference examples)"]:::surface
    P2["**Phase 2**<br/>SDK plugin, in-process<br/>(flyteplugins-skypilot)"]:::surface
    P3["**Phase 3**<br/>Async connector deployment<br/>(flyte-connector-skypilot)"]:::bridge
    P4["**Phase 4**<br/>Typed IDL surface<br/>(flyteidl2.plugins.skypilot)"]:::bridge
    P5["**Phase 5**<br/>SkyServeAppEnvironment<br/>(app/serving plane)"]:::deep
    P6["**Phase 6**<br/>Devbox SkyPilot backend<br/>(CLI + interactive)"]:::deep
    P7["**Phase 7**<br/>Lineage, cost &amp; telemetry<br/>(backend + UI)"]:::deep
    P8["**Phase 8**<br/>Runtime cluster registration<br/>(FlyteAdmin API / operator webhook)"]:::h2
    P9["**Phase 9**<br/>Pre-baked data plane image +<br/>ephemeral data plane lifecycle"]:::h2

    P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7 --> P8 --> P9

    L1[/"Horizon 1 — SDK layer<br/>(zero data-plane changes)"\]:::surface
    L2[/"Horizon 1 — Bridge layer<br/>(new deployment, typed IDL)"\]:::bridge
    L3[/"Horizon 1 — Control-plane layer<br/>(devbox, UI, backend)"\]:::deep
    L4[/"Horizon 2 — Ephemeral data planes<br/>(dynamic registration + bootstrap)"\]:::h2

    L1 -.-> P1
    L1 -.-> P2
    L2 -.-> P3
    L2 -.-> P4
    L3 -.-> P5
    L3 -.-> P6
    L3 -.-> P7
    L4 -.-> P8
    L4 -.-> P9
```

### 8.2 What each phase ships, and why it sequences this way

#### Horizon 1 — connector plugin

| Phase | Layer          | Deliverable                                       | What gets de-risked for the next phase                                                                                                       | Effort |
|-------|----------------|---------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|--------|
| 1     | SDK            | Reference examples: `sky launch` from a normal Flyte task via `ContainerTask` or shell-out. No new SDK code. | Validates demand with design partners and surfaces authoring frictions before any plugin code is committed. Lowest commitment. | S      |
| 2     | SDK            | `flyteplugins-skypilot` with `SkyPilot` / `SkyPilotJob` dataclasses and `SkyPilotTask`, running the SkyPilot Python SDK **in-process** from the user's task pod (`AsyncConnectorExecutorMixin`). | Locks in dataclass shape, `sky.Task` round-trip serialization, secrets injection, and authoring ergonomics — without committing to a new deployment topology. | M      |
| 3     | Bridge         | Promote to a true async connector with its own K8s deployment (`flyte-connector-skypilot`), matching the Databricks/BigQuery topology. Also validates that SkyPilot's Postgres-backed HA API server is stable under team-shared load. | Decouples failure boundaries, centralizes credentials, removes "every task pod ships SkyPilot" cost, proves connector scales under fanout. | M      |
| 4     | Bridge         | `flyteidl2.plugins.skypilot` proto so launch specs are typed end-to-end. | Gives the backend a stable schema to introspect — prerequisite for UI surfacing of SkyPilot intent and for Horizon 2 cluster registration. | S      |
| 5     | Control plane  | `SkyServeAppEnvironment` integrated with the Flyte app/serving lifecycle, including replica/autoscaling translation. | First control-plane-aware surface. Validates connector + app can co-exist before touching devbox or telemetry. | M      |
| 6     | Control plane  | `flyte devbox launch --plugin skypilot ...` boots a SkyPilot-managed pod; CLI handles SSH tunnel, code sync, teardown. | Most invasive CLI work; gated on phases 2–5 and on research-team demand for SkyPilot-style SSH interactivity. | L      |
| 7     | Control plane  | Cost, spot-recovery, and multi-cluster routing telemetry in the Flyte UI and run graph. Optional Flyte queue ↔ SkyPilot priority affinity. | Pure deepening. Defer until adoption justifies the UI/backend investment. | L      |

#### Horizon 2 — ephemeral data planes

| Phase | Layer                   | Deliverable                                                                                               | What gets de-risked                                                                                                   | Effort |
|-------|-------------------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|--------|
| 8     | Control plane / infra   | Runtime cluster registration API in FlyteAdmin (or Union operator webhook): an ephemeral data plane registers itself on boot and deregisters on drain without a control-plane Helm restart. | This is the key missing primitive. Gates all of Horizon 2. | L      |
| 9     | Infra / platform        | Pre-baked Union data plane VM image (AMI / GCP image): k3s + FlytePropeller + Union operator pre-installed, so bootstrap is a cluster join and Helm upgrade (~2–5 min) rather than a full install. SkyPilot provisions VMs from this image, bootstraps the data plane, and the Union operator phones home to register with the control plane. | Closes the end-to-end Horizon 2 loop: SkyPilot provisions → k3s bootstraps → Union data plane registers → work runs → autostop → deregister. | L      |

Effort uses T-shirt sizing (S ≈ ~2 eng-weeks, M ≈ ~8 eng-weeks, L ≈ ~16 eng-weeks). Horizon 2 phases are gated on Horizon 1 phases 2–4 being stable and on explicit product direction to build dynamic cluster registration.

### 8.3 Decision points between phases

**Horizon 1:**
- **After phase 1.** Did at least two design partners adopt the subprocess pattern? If no, insufficient demand to justify continuing.
- **After phase 2.** Does the in-process SDK feel right ergonomically? Are users hitting failure modes that a dedicated connector deployment would fix? If yes, proceed to phase 3.
- **After phase 4.** Is the typed IDL stable and does the connector hold up under realistic load? Phases 5–7 are gated on a yes here.
- **Before phase 6.** Is devbox usage growing? Only tackle when there is a clear pull from researchers asking for SkyPilot-style interactivity.

**Horizon 2:**
- **Before phase 8.** Is the Horizon 1 connector stable and in production use with at least one design partner running multi-stage factory workloads? Horizon 2 is motivated by the operational cost of always-on data plane clusters — the pain must be demonstrated, not assumed.
- **After phase 8.** Does the dynamic registration API work reliably under concurrent register/deregister cycles? Phase 9 depends on this being rock-solid, since a registration race during bootstrap would orphan a running data plane.

This gating structure is the explicit answer to "how do we de-risk deeper engineering work": **each phase is independently shippable, each phase is gated on validation from the prior, and Horizon 2 is intentionally the last thing we commit to.**

---

## 9. Risks & open questions

### 9.1 Maintainer-side risks

- **SkyPilot SDK churn.** SkyPilot ships frequently and has occasionally renamed Python APIs. Mitigation: pin to a tested minor range, vendor the small subset we depend on into a thin adapter in `_models.py`, and run a daily integration test against SkyPilot's `latest`.
- **Connector deployment scaling.** Long-running pollers can hog the connector pod under heavy fanout. Mitigation: switch the SkyPilot connector to event-driven polling (use SkyPilot's webhook/event API once it stabilizes) before we encourage 1000-fanout patterns.
- **Code-bundle vs `workdir` divergence.** Flyte ships code via the code bundle; SkyPilot ships code via `workdir`. We must pick one as authoritative and document the contract. Default: use the Flyte code bundle, copy it into the SkyPilot setup so `~/sky_workdir` matches what Flyte sees.
- **Cost attribution.** SkyPilot's API server emits cost telemetry; Flyte's UI does not currently render it. Mitigation: surface it as `Link` outputs in the surface/bridge layers, build dedicated UI in the control-plane layer.
- **[Horizon 2] Dynamic registration correctness.** A register/deregister race during a rolling bootstrap could orphan a data plane or route work to a cluster mid-teardown. Mitigation: registration must be gated on a readiness probe (Propeller healthy + shared object store accessible) and deregistration must drain in-flight work before removing the cluster config.
- **[Horizon 2] Shared object store access.** All Flyte data planes must write to the same S3/GCS bucket as the control plane — this is a known constraint from the existing multi-data-plane deployment guide. SkyPilot's credential forwarding naturally gives the ephemeral cluster access; this must be validated in the bootstrap playbook.

### 9.2 Open questions

1. **Single vs. per-team SkyPilot API server**: A single Postgres-backed HA server simplifies operations; per-team servers improve blast-radius isolation. The choice shapes both the auth model and how the connector discovers which API server to call.
2. **Who operates the SkyPilot API server**: Self-operated, or a Union-hosted managed add-on? The latter is a natural SKU given Union already operates managed devbox infrastructure.
3. **Horizon 2 registration API design**: Should the runtime cluster registration surface be a new FlyteAdmin gRPC endpoint, a Kubernetes CRD that FlyteAdmin watches, or a webhook on the Union operator? Each has different security and blast-radius trade-offs.
4. **Bootstrap latency target**: What is the maximum tolerable cold-start time before the model factory would prefer a warm-standby cluster? The answer determines whether a pre-baked image is sufficient or whether a warm-pool strategy is needed.
5. **SkyPilot vs. Armada for multi-cluster routing**: For organizations operating multiple permanent BYOC data planes, does [Armada](https://armadaproject.io/)'s multi-cluster job scheduling add value on top of FlyteAdmin's existing `labelClusterMap` routing? Evaluate as the cluster fleet grows; not a dependency for either horizon.
6. **Billing accountability**: Cloud compute for SkyPilot-provisioned capacity lands on the customer's cloud bill. For Union-hosted SkyPilot API server, metering must be based on provisioned GPU-hours (SkyPilot already emits this telemetry), not on Flyte task count.

---

## 10. Value proposition and positioning

### 10.1 Where this fits

This integration addresses organizations that are missing one or both of:

- **Orchestration + lineage** on top of existing SkyPilot workflows (Argo/Airflow subprocesses, hand-rolled polling, no typed I/O).
- **Zero-idle multi-cloud compute** on top of an existing Union/Flyte deployment (fixed K8s cluster pool, no spot bidding across clouds).

**Common entry points:**

- Already runs SkyPilot in dev/research but lacks a durable pipeline layer.
- Already runs Union/Flyte but needs to reach spot GPU capacity outside the home cluster.
- Has a "model factory" pattern — continuous fine-tune → eval → deploy — that runs manually today.
- GPU spend with explicit spot/multi-region cost pressure.

### 10.2 Value-prop framing

When the team already runs SkyPilot:

> *"Keep SkyPilot for finding cheap, available GPUs across your clouds. Add Flyte 2 on top for the durable pipelines, lineage, and checkpoint-triggered evals that SkyPilot deliberately does not provide. Same Python authoring experience; existing SkyPilot YAML still works underneath."*

When the team already runs Flyte / Union:

> *"You already have the pipeline, lineage, and trigger layer. The SkyPilot plugin lets that pipeline reach across regions and clouds for spot capacity without giving up the audit trail — and, with Horizon 2, without maintaining standing GPU clusters between factory runs."*

### 10.3 Deployment options

- **Plugin (Horizon 1) is OSS** (Apache 2.0, in the `flyte-sdk` monorepo) — consistent with all `flyteplugins-*` packages.
- **Hosted SkyPilot API server**: Teams who do not want to operate their own Postgres-backed API server can use a Union-managed add-on. Union already operates SkyPilot infrastructure for managed devbox; this is an extension of that.
- **Horizon 2 ephemeral data planes**: Self-operated (bring your own cloud credentials + pre-baked image), or Union-managed (Union handles the bootstrap playbook and VM image lifecycle as a platform service).
- **Multi-cloud cost dashboards** (phase-7 UI work) are a natural Union-only feature.
- **Support scope**: Connector, IDL, SDK, and best-effort triage of upstream SkyPilot issues. Upstream fixes go through the SkyPilot OSS project.

### 10.4 Competitive positioning

| Alternative                      | Gap                                                                                | How the layered stack answers it                                                  |
|----------------------------------|------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| Anyscale (Ray + serve)           | Single-runtime lock-in                                                             | Flyte keeps Ray, Spark, vLLM, SkyPilot on equal footing                          |
| Dagster + sky subprocess         | Broken lineage at the DE→ML handoff; no typed I/O or semantic replay               | Single Flyte graph from ETL through training, eval, and serve                     |
| Armada (multi-cluster routing)   | Routes across pre-existing always-on clusters — idle cost remains                  | SkyPilot Horizon 2 creates and destroys clusters; zero idle cost                  |
| Pure Flyte today                 | No multi-cloud failover, no spot recovery, no sky serve, no zero-idle data planes  | Plugin closes all four gaps without core rewrites                                 |
| Standalone SkyPilot              | No DAGs, lineage, typed I/O, semantic caching, or pipeline durability              | Flyte wraps SkyPilot compute with the full factory control plane                  |

---

## 11. Appendix

### 11.1 Conformance with existing plugin conventions

| Convention                                                  | Where                                            | This plugin                                  |
|-------------------------------------------------------------|--------------------------------------------------|----------------------------------------------|
| Package name `flyteplugins-<name>`                          | `plugins/dask/pyproject.toml`                    | `flyteplugins-skypilot`                      |
| Public exports via `__init__.py`                            | `plugins/ray/src/flyteplugins/ray/__init__.py`   | `SkyPilot`, `SkyPilotJob`, `SkyPilotConnector`, `SkyServeAppEnvironment` |
| `@dataclass` config passed via `plugin_config=`             | `flyteplugins.dask.Dask`                         | `flyteplugins.skypilot.SkyPilot`             |
| Subclass for richer variant                                 | `Databricks(Spark)`                              | `SkyPilotJob(SkyPilot)`                      |
| `AsyncConnector` + `ConnectorRegistry.register(...)`        | `flyteplugins.databricks.DatabricksConnector`    | `flyteplugins.skypilot.SkyPilotConnector`    |
| `AsyncConnectorExecutorMixin` for local exec                | `flyteplugins.databricks.DatabricksFunctionTask` | `flyteplugins.skypilot.SkyPilotTask`         |
| `AppEnvironment` subclass for services                      | `flyteplugins.vllm.VLLMAppEnvironment`           | `flyteplugins.skypilot.SkyServeAppEnvironment`|
| `flyteidl2.plugins.<name>_pb2` for typed custom-config      | `flyteidl2.plugins.spark_pb2`                    | `flyteidl2.plugins.skypilot_pb2` (bridge layer) |
| Connector secrets via `kubectl set env deployment/...`      | Connector docs in `integrations.md`              | `SKYPILOT_API_SERVER_ENDPOINT`, `skypilot_token` |

### 11.2 References

**SkyPilot:**
- SkyPilot docs: <https://docs.skypilot.co/en/latest/docs/index.html>
- SkyPilot YAML spec: <https://docs.skypilot.co/en/latest/reference/yaml-spec.html>
- SkyPilot API server: <https://docs.skypilot.co/en/latest/reference/api-server/api-server.html>
- SkyPilot API server HA + Postgres: <https://docs.skypilot.co/en/latest/reference/api-server/api-server-upgrade.html>
- SkyPilot state management: <https://docs.skypilot.co/en/pool-docs/reference/architecture/state.html>
- SkyPilot on existing machines (k3s bootstrap): <https://docs.skypilot.co/en/latest/reservations/existing-machines.html>

**Flyte / Union:**
- Flyte 2 user guide: <https://www.union.ai/docs/v2/flyte/user-guide/>
- Flyte 2 integrations: <https://www.union.ai/docs/v2/flyte/integrations/>
- Flyte 2 plugin source: <https://github.com/flyteorg/flyte-sdk/tree/main/plugins>
- Flyte multi-data-plane deployment: <https://github.com/flyteorg/flyte/blob/master/docs/deployment/deployment/multicluster.rst>
- Union BYOC platform architecture: <https://www.union.ai/docs/v2/union/deployment/byoc/platform-architecture/>

**Other:**
- Armada multi-cluster job scheduling: <https://armadaproject.io/>
- Companion specs: [`SPEC.md`](./SPEC.md) (FLYRES stack), [`COMPARISON.md`](./COMPARISON.md) (Flyte vs Dagster)

### 11.4 Out of scope (explicitly deferred)

- Rewriting SkyPilot's API server in Go to embed it in the Flyte data plane. Not necessary; not aligned with the connector pattern.
- Cost-based scheduling decisions made by Flyte (vs. SkyPilot). SkyPilot owns the bidder; Flyte owns the trigger.
- Slurm-as-target. SkyPilot supports it; we will validate but not productize until a design partner requests it.
- Ray-on-SkyPilot composition (`flyteplugins.ray` + `flyteplugins.skypilot` on the same task). Possible but a phase-4 conversation.
- Armada as a replacement for Horizon 2. Armada routes across pre-existing clusters; SkyPilot creates and destroys them. They solve different problems and are evaluated independently.

---

## 12. Horizon 2: Ephemeral Union Data Planes

This section is a design sketch for Horizon 2, intended to capture the architectural vision and the two remaining engineering gaps. It does not constitute a committed spec until phase 8 is explicitly greenlit.

### 12.1 Vision

> *"SkyPilot could launch a Union cluster anywhere."*
> — Ketan Umare, Union CEO

Today, a Union data plane is a Kubernetes cluster that is always running, registered statically with the Union control plane, and paid for whether or not it is executing work. In Horizon 2, SkyPilot provisions the VMs, bootstraps the K8s cluster, installs the data plane, and registers it — all in response to a Flyte run being submitted that requires more capacity than is currently registered. When the run completes and the cluster is idle, SkyPilot autostops it and the Union operator deregisters it. The control plane sees a normal data plane for the duration of the run and no data plane when idle.

### 12.2 Architecture

```
Union Control Plane (always-on, hosted)
      │
      │  ← runtime registration (phase 8)
      │
      ▼
SkyPilot API Server (Postgres-backed HA)
      │
      │  sky launch --image union-data-plane-ami
      │
      ▼
Cloud VMs (spot, multi-region)
      │
      │  sky local up --ips <provisioned VMs>
      │
      ▼
k3s cluster
      │
      │  helm install flyte-core (propeller + union operator)
      │
      ▼
Union Data Plane (ephemeral)
      │
      │  union-operator phones home → registers with control plane
      │
      ▼
Work executes with full Flyte lineage (shared object store)
      │
      │  SkyPilot autostop → union-operator drains + deregisters
      │
      ▼
Cluster destroyed. Control plane has the lineage; the cluster is gone.
```

### 12.3 What already works

| Component | Status | Notes |
|-----------|--------|-------|
| SkyPilot Postgres-backed HA API server | ✅ Available | Resolves original state-persistence concern |
| `sky local up --ips …` k3s bootstrap | ✅ Available | Deploys k3s + GPU Operator from any SSH-accessible VMs |
| Flyte multi-data-plane `clusterConfigs` routing | ✅ Available | FlyteAdmin can route to multiple registered data planes |
| Union BYOC operator (data plane lifecycle) | ✅ Available | Manages data plane from within the cluster |
| SkyPilot credential forwarding | ✅ Available | Temporarily forwards cloud creds to provisioned nodes; gives ephemeral cluster access to shared object store |

### 12.4 What needs to be built

**Gap 1 — Runtime cluster registration (phase 8, effort L):**

FlyteAdmin currently requires static `clusterConfigs` in Helm values; adding or removing a data plane requires a pod restart. The Union operator needs a way to register a new cluster at runtime — either:

- A new gRPC/REST endpoint on FlyteAdmin that accepts a signed cluster credential bundle.
- A Kubernetes CRD (`FlyteclusterRegistration`) in the control plane namespace that FlyteAdmin watches and reconciles into its routing table.
- A webhook on the Union operator that calls FlyteAdmin on pod startup and pod drain.

The registration must be gated on a readiness signal (Propeller healthy + shared object store accessible + GPU Operator ready) to prevent routing work to a mid-bootstrap cluster.

**Gap 2 — Pre-baked data plane VM image (phase 9, effort L):**

Cold install of k3s + Helm + FlytePropeller + Union operator + NVIDIA GPU Operator on a blank VM takes 15–30 minutes. That latency is acceptable for long-running training runs but not for short evaluation jobs. A pre-baked AMI (or GCP image) with all dependencies pre-installed reduces bootstrap to a cluster join + Helm upgrade (2–5 minutes). This image must be:

- Maintained as a versioned artifact alongside each Union release.
- Built for each major cloud (AWS AMI, GCP image, Azure image) and each GPU hardware generation.
- Tested daily against SkyPilot's `sky local up` bootstrap path.

### 12.5 Data flow in Horizon 2

The shared object store is the anchor for lineage consistency across both horizons:

- Inputs are materialized to the shared S3/GCS bucket before the ephemeral data plane boots.
- The ephemeral data plane reads inputs from and writes outputs to the same bucket using credentials forwarded by SkyPilot.
- FlyteAdmin's lineage graph references URIs in the shared bucket — identical to how it works with a permanent data plane.
- After the cluster is destroyed, the full run history is queryable from the Union control plane UI as if it had run on a permanent cluster.

### 12.6 Relation to Armada

[Armada](https://armadaproject.io/) routes job queues across multiple pre-existing, always-on Kubernetes clusters, making multi-cluster scheduling transparent to submitters. It solves a different problem:

- **Armada**: _I have N always-on clusters; route each job to whichever has available capacity._
- **SkyPilot Horizon 2**: _I have zero standing clusters; provision one on the cheapest available spot capacity for this run, then destroy it._

These are complementary patterns. An organization with a mix of reserved BYOC data planes (Armada) and burst workloads (SkyPilot Horizon 2) could use both. Neither is a prerequisite for the other, and both integrate with the same Union control plane.

