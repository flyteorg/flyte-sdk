# FLYRES Stack Specification

**FLYRES** (Flyte + Ray + SkyPilot) is a reference stack for running AI workloads on Kubernetes.

## Context

This spec describes the requirements of an AI research organization that needs to support
both foundation-model research (distributed training, fine-tuning, evaluation) and the
production serving of those models to customer-facing endpoints. The organization also
operates a data engineering function whose primary purpose is to support AI research
(e.g. ingesting and curating large multimodal training datasets).

The target stack is intended to consolidate the orchestration, compute, and infrastructure
layers across both research and production while minimizing the number of tools researchers
and platform engineers need to learn and operate.

## Target Workloads

- **Distributed training**: Pretraining and fine-tuning of large models using frameworks
  such as FSDP, DeepSpeed, and Megatron.
- **Model serving**: Customer-facing inference endpoints, typically built on top of
  llm-d-style serving runtimes.
- **Data engineering pipelines**: Large-scale ingestion and preprocessing of multimodal
  datasets (e.g. downloading millions of hours of media files, transcoding, filtering,
  and curating training datasets).
- **Interactive research**: Researchers iterating on training code, experimenting with
  new libraries, CUDA versions, and hardware.

## Personas

- **AI researchers**: Want a low-friction, SSH-like development experience. Resist tools
  that require restructuring their code into orchestrator-specific abstractions. Need to
  pin specific GPU instances when iterating on long-running training runs.
- **Data engineers**: Need a pipeline orchestrator for data preprocessing and ETL that
  feeds into research datasets.
- **Platform / infra engineers**: Operate the compute fleet across multiple Kubernetes
  clusters, manage shared storage, and own observability, tracking, and reproducibility.
- **Leadership**: Wants better visibility into what changes between model weights and
  serving runtime across releases, especially after incidents.

## Constraints and Pain Points

1. **Scarce GPU capacity**: A relatively small fleet of GPU nodes is shared across many
   researchers. Researchers need the ability to pin specific instances for the duration
   of a run because contention is high.
2. **Data gravity**: Models, datasets, and container images are large. Moving model
   weights into all nodes can take an hour or more via rsync. The stack must avoid
   gratuitous data movement and exploit locality (shared storage, caching, hf-mounts,
   etc.).
3. **Multi-cluster Kubernetes**: Compute spans multiple Kubernetes clusters. Dev and
   prod workloads share the same underlying GPU fleet and must be scheduled across
   clusters seamlessly.
4. **Kubernetes lifecycle**: The organization wants its Kubernetes clusters installed,
   upgraded, and managed by tooling rather than operated as a set of bare nodes.
5. **Container build times**: Building training images with CUDA-dependent packages can
   take an hour or more per build. Iteration on new CUDA versions, new chips, or new
   libraries is painfully slow. The stack should optimize image build times and reuse.
6. **Reproducibility and tracking**: A previous failed launch exposed gaps in tracking
   what changed between releases, particularly the coupling of model weights and serving
   runtime. The stack must provide first-class versioning and lineage across training,
   artifacts, and serving runtimes.
7. **Tool consolidation**: The organization wants to avoid running separate orchestrators
   for AI research and for data engineering. A single orchestrator should be capable of
   serving both teams.

## Required Capabilities

The stack must, at minimum, provide:

- **Macro orchestration** of end-to-end workflows that span data prep, training,
  evaluation, and deployment.
- **Distributed compute** primitives suitable for large-scale training and inference.
- **Multi-cluster scheduling** across multiple Kubernetes clusters with shared GPU pools
  between dev and prod.
- **Model and dataset storage** with mountable shared storage so that large artifacts do
  not have to be re-downloaded per node.
- **Experiment tracking** for metrics, hyperparameters, and run metadata.
- **Artifact lineage and versioning** that ties model weights to the exact training run
  and serving runtime in which they are deployed.
- **Interactive / SSH-style developer experience** so that researchers can iterate
  without rewriting their code into orchestrator-specific abstractions.
- **Efficient container image builds** for CUDA-heavy training images, with caching and
  reuse across builds.
- **Inference serving** for customer-facing endpoints, ideally co-located with the same
  GPU fleet used for training.

## Reference Stack

The reference stack chosen to satisfy these requirements is:

- **Ray**: Distributed compute runtime for training and large-scale parallel workloads.
- **Flyte / Union**: Macro orchestrator for both AI research workflows and data engineering
  pipelines; owns lineage, versioning, and tracking across the full lifecycle. Union's
  managed offering separates the control plane (hosted by Union) from the data plane
  (running in the customer's cloud account).
- **SkyPilot**: Multi-cloud compute broker and cluster bootstrapper. In the near-term
  (Horizon 1), SkyPilot runs individual tasks on behalf of Flyte via a connector plugin,
  providing spot recovery and multi-region failover. In the longer term (Horizon 2),
  SkyPilot bootstraps entire ephemeral Union data planes on demand — provisioning VMs,
  standing up k3s, installing FlytePropeller, and registering the cluster with the Union
  control plane — so that compute scales to zero when idle and is sourced from wherever
  capacity is cheapest at the time a run is triggered. See `SPEC_SKYPILOT_INTEGRATION.md`
  for the full phasing.
- **Armada** (optional, later): Routes work across pre-existing, always-on Kubernetes
  clusters when reserved capacity matters more than zero-idle cost. Complementary to
  SkyPilot: Armada for reserved BYOC clusters, SkyPilot for elastic burst. Evaluated
  separately; not a blocking dependency for Horizon 1 or 2.
- **Hugging Face**: Model storage and shared-storage mounts (hf-mounts) for large
  artifacts.
- **Weights & Biases**: Experiment tracking, with the option to replace it over time
  with a more cost-effective tracking layer integrated into the orchestrator.

## Flyte vs. Dagster: Unbiased Comparison

This section provides an objective comparison between Flyte 2 and Dagster, two popular
orchestration frameworks for ML workflows.

**Purpose**: The goal is to help teams evaluate both options based on their specific
requirements, rather than declaring a winner. Both tools have valid use cases.

**Methodology**: Comparison based on:
- Official documentation (flyte.org, dagster.io)
- Open-source code review
- Community adoption patterns
- Feature parity analysis

**Updates**: Last updated for Flyte 2.x and Dagster 1.7+.

### Architecture Overview

| Aspect | Flyte 2 | Dagster |
|--------|---------|--------|
| **Orchestration Model** | Task-centric: All workloads are tasks with inputs/outputs; DAGs are implicit from function dependencies | Asset-centric: Data and code as first-class citizens; DAGs explicit via job/solid definitions |
| **Execution Environment** | Flexible: Can run locally (devbox), in cloud K8s, or delegate to external schedulers like SkyPilot. No container build required for development. | Flexible: Can run locally, in containers, or on cloud infra. K8s support exists but is not first-class. |
| **Resource Management** | Fine-grained per-task resource requests (cpu/memory/gpu) with native Kubernetes integration | Resource constraints via op/asset decorators; custom executors required for advanced scheduling |
| **Scaling Model** | Horizontal scaling per task via plugins (Ray, Dask); container-based parallelism | Parallelism via queue-based executors; requires external tools for distributed compute |

### Core Capabilities Comparison

| Capability | Flyte 2 | Dagster | Notes |
|------------|---------|--------|------|
| **Workflow Definition** | Python functions with decorators; implicit DAGs from function dependencies. Supports `///` script headers for uv-based workflows without Docker. | Ops/solids + graphs/jobs; explicit graph construction via composable functions | Flyte feels more natural for code-first workflows; Dagster clearer for data pipeline visualization |
| **Scheduling** | Cron triggers (`flyte.Trigger.cron()`); manual, hourly, daily presets with timezone support | `@schedule` decorators with cron and sensor triggers | Both support similar scheduling primitives |
| **Data Handling** | Typed outputs/inputs; automatic blob storage transfers between tasks. Uses `flyte.File`, `flyte.Dir`, etc. | Assets with metadata + I/O managers for data persistence | Flyte handles data movement transparently; Dagster requires explicit I/O manager config |
| **Caching** | Semantic caching by task hash + inputs; configurable policy per task | Op-level output caching via decorators with manual cache key configuration | Flyte's semantic caching is more automatic and robust |
| **Retry Logic** | Configurable per-task; exponential backoff with max retries | Retry strategy on op/asset execution | Similar capabilities in both |
| **Versioning** | Task code versioned alongside executions via code bundle + image hash. Replayable historical runs with exact dependencies. | Asset/node definitions tracked; lineage available but replay limited to current code version | Flyte has stronger reproducibility guarantees for production workloads |

### ML-Specific Features

| Feature | Flyte 2 | Dagster | Notes |
|---------|---------|--------|------|
| **Distributed Training** | Native plugins: Ray, PyTorch Elastic, TensorFlow. FSDP and DDP via `flyteplugins.pytorch.task.Elastic`. No container rebuild needed during development (code bundle). | Requires custom executors or external tools (Ray, Dask) for distributed training |
| **GPU Management** | Per-task GPU requests (`Resources(gpu="A10G:1")`); native Kubernetes GPU scheduling. Supports multi-GPU and multi-node via plugins |
| **Model Registry Integration** | First-class model versioning examples. Can integrate with MLflow, Weights & Biases, or custom registries via tasks |
| **Experiment Tracking** | Easy integration with W&B, MLflow, etc. via tasks. No special setup required. |
| **Development Experience** | `///` script headers + uv support. Run locally without Docker using `flyte.with_runcontext(mode="local")`. Code travels as bundle during development. | Python-first but typically requires containerization for K8s deployments. More manual setup for local dev. |

### Operational Characteristics

| Concern | Flyte 2 | Dagster | Notes |
|---------|---------|--------|------|
| **Deployment Complexity** | Devbox mode: run locally without Docker. Production: K8s deployment via `flyte.deploy()`. Image builds are incremental (layered). | Local dev easy; K8s deployment requires custom executor configuration. Less documentation for production deployments. |
| **Multi-Cluster Support** | Plugin-based; can delegate to external schedulers (e.g., SkyPilot). Union.ai offers multi-cluster support as a managed service. | Limited native support; requires custom executors and careful architecture |
| **Developer Experience** | Pythonic: `///` script headers with uv dependencies. Local execution via `flyte.with_runcontext(mode="local")`. Code bundle avoids container rebuilds during development. | Python-first but typically requires containers for K8s. More manual setup for local testing without Docker. |
| **Observability** | Built-in UI with task-level logs, metrics, lineage graph. Full audit trail of all executions. | Webserver with asset/task views; lineage via metadata tracking |
| **Community & Ecosystem** | Strong in AI/ML space; growing adoption. Open source under Apache 2.0. Union.ai provides commercial support. | Larger general-purpose ecosystem; enterprise adoption higher in data engineering. Dagster Labs provides commercial support. |

### When to Choose Flyte 2

Choose Flyte if:
- You want Pythonic workflows without Docker for development (`///` script headers + uv)
- Your workloads are heavily distributed (Ray, PyTorch Elastic, etc.)
- You need strong lineage tracking and reproducibility guarantees
- GPU scheduling and management is a key requirement
- You're building an ML platform with model registry integration
- You prefer container-based deployment for consistency across environments

### When to Choose Dagster

Choose Dagster if:
- Your team already uses Dagster for data engineering
- You have more traditional ETL/data engineering workloads
- You need local development without any containerization concerns
- You want a more asset-centric mental model for data pipelines
- You already use the Python data ecosystem (pandas, pyarrow) heavily and prefer its patterns
- Enterprise features and commercial support from Dagster Labs are important considerations

### Migration Considerations

**From Dagster to Flyte:**
- Existing op logic can often be migrated with minimal changes (functions become tasks)
- Scheduling requires updating from `@schedule` to `flyte.Trigger.cron()` or other triggers
- I/O managers need replacement with Flyte's typed data system (`flyte.File`, `flyte.Dir`)
- Local development uses `flyte.with_runcontext(mode="local")` instead of container builds
- Code bundles avoid Docker rebuilds during development (faster iteration)

**From Flyte to Dagster:**
- Tasks can become ops with minor signature changes
- Distributed training requires adding external tools (Ray, Dask) as executors
- Data lineage tracking may need custom metadata plugins
- GPU scheduling requires custom executor configuration
- Local dev becomes easier (no Docker at all), but production K8s deployments require more setup
- Tasks can become ops with minor signature changes
- Distributed training requires adding Ray/Dask executors
- Data lineage tracking may need custom metadata plugins
- GPU scheduling requires custom executor configuration

## Flyte ↔ SkyPilot Integration

The integration is structured across two horizons. Both are forward-looking; neither
requires forking SkyPilot or changing Flyte's core. See `SPEC_SKYPILOT_INTEGRATION.md`
for the full phase-by-phase engineering spec.

### Architecture note: SkyPilot's matured API server

SkyPilot previously stored all state in local files on the machine running `sky`,
making it awkward to operationalize in shared-team settings. That constraint has since
been resolved: SkyPilot's [API server](https://docs.skypilot.co/en/latest/reference/api-server/api-server.html)
is now a separately deployed service that supports an external PostgreSQL database for
state persistence, rolling upgrades (zero-downtime), and Helm-based HA deployment.
This architectural split is what makes both integration horizons below viable.

### Tool boundaries

Flyte and SkyPilot operate at genuinely different abstraction layers and are
complementary rather than competing:

- **SkyPilot is a compute control plane.** It provisions VMs or Kubernetes pods on
  20+ clouds plus Kubernetes/Slurm, handles spot/preemptible recovery, autostop, and
  idle teardown, and exposes a centralized API server for team-shared access. SkyPilot
  can also bootstrap lightweight Kubernetes clusters (`sky local up --ips …` deploys
  k3s on any list of SSH-accessible machines).
- **Flyte / Union is a workflow orchestration and durable execution platform.** It owns
  the DAG, typed I/O, data lineage, semantic caching, retries, and the full audit trail
  of what ran, with what inputs, and what came out. Union's control/data plane split
  means a new execution environment is just a Kubernetes cluster registered with
  FlyteAdmin — it does not require touching the control plane.

### Horizon 1: connector plugin (near-term)

SkyPilot runs individual Flyte tasks on multi-cloud infrastructure via a first-class
`flyteplugins-skypilot` connector. Flyte remains the orchestrator; SkyPilot is a
compute backend that individual tasks can target.

**Best practices in this model:**

1. **Flyte for orchestration, SkyPilot for compute**: Flyte manages workflow logic,
   retries, and lineage; SkyPilot handles GPU scheduling, spot recovery, and
   multi-region failover.
2. **Shared object store as the data contract**: SkyPilot tasks write outputs to the
   same S3/GCS bucket Flyte uses. `flyte.File` / `flyte.Dir` references are the typed
   handoff between stages — no rsync, no opaque paths.
3. **Flyte for data prep, SkyPilot for GPU stages**: Preprocessing and ETL run in the
   home cluster pool; GPU-intensive training and evaluation target SkyPilot with spot
   recovery and managed job failover.
4. **Connector as the single failure boundary**: Only `flyte-connector-skypilot` sits
   between Flyte Propeller and the SkyPilot API server. Propeller never blocks on
   SkyPilot availability directly.

**What Flyte adds on top of raw SkyPilot:**

- **Semantic caching**: Flyte knows whether a preprocessing step has already run on
  this exact dataset version and skips it. SkyPilot does not cache at that semantic
  level.
- **Data lineage**: Every artifact (dataset version, checkpoint, eval score) is typed
  and tracked through the run graph. When a model behaves differently in prod, the
  lineage traces back to the exact data and training run that produced it.
- **Durable execution**: If a node fails mid-pipeline, Flyte resumes from the last
  successful task. SkyPilot handles job-level retries but not pipeline-level
  resumption.
- **Versioned reproducibility**: Flyte versions task code alongside executions, so any
  historical pipeline can be re-run exactly.

### Horizon 2: ephemeral Union data planes (longer-term)

Rather than running individual tasks on SkyPilot infrastructure, SkyPilot bootstraps
an entire Union data plane on demand — provisions cloud VMs, stands up k3s, installs
FlytePropeller and the Union operator, and registers the cluster with the Union control
plane. When the workload finishes, SkyPilot autostops and the cluster deregisters. The
Union control plane never restarts; the data plane is entirely ephemeral.

This closes the original gap Ketan identified: _"SkyPilot could launch a Union cluster
anywhere"_ — any cloud, any region, using spot capacity, with zero idle cost between
factory runs.

**What already exists to support this:**

- `sky local up --ips <hosts>` deploys k3s on any list of SSH-accessible VMs, installs
  the GPU Operator, and configures kubeconfig — the entire cluster bootstrap in one
  command.
- Flyte's `clusterConfigs` + `labelClusterMap` in FlyteAdmin already routes work to
  multiple registered data planes.
- SkyPilot's credential-forwarding mechanism (it temporarily forwards cloud credentials
  to provisioned nodes) naturally gives the ephemeral cluster access to the shared
  object store without Flyte ever seeing the raw credentials.
- The Union BYOC operator model already manages the data plane lifecycle from within the
  cluster.

**Two gaps to close:**

1. **Dynamic cluster registration**: FlyteAdmin currently requires static
   `clusterConfigs` in Helm, with a pod restart to add or remove a cluster. A runtime
   registration API (or a Union operator webhook that phones home on boot and
   deregisters on drain) is needed so ephemeral clusters can self-register.
2. **Bootstrap latency**: k3s + FlytePropeller + Union operator install time should be
   compressed to 2–5 minutes by using a pre-baked VM image (AMI / GCP image) with all
   dependencies present, so bootstrap is a cluster join and Helm upgrade rather than a
   full install from scratch.

### SkyPilot vs. Armada

[Armada](https://armadaproject.io/) takes a different approach to the multi-cluster
problem: it routes work across pre-existing, always-on Kubernetes clusters, making
multi-cluster scheduling transparent to job submitters. The two tools are complementary
rather than competing:

- **Armada**: best for reserved or on-prem BYOC clusters that are always running and
  need intelligent cross-cluster bin-packing.
- **SkyPilot**: best for elastic burst capacity that should cost nothing when idle —
  provisions clusters on demand and tears them down when work is done.

For the FLYRES stack, SkyPilot is the more immediately useful choice because the
dominant pain point is GPU scarcity and cost, not scheduling across a large fleet of
always-on clusters. Armada is worth tracking for the case where the organization
operates multiple permanent BYOC data planes and wants Union-level visibility across
all of them.

### Data gravity

Data gravity is the strongest differentiator for an AI-native orchestrator in this
stack. Long rsync times, hour-long Docker builds for CUDA packages, and large
checkpoints are all data gravity problems. Flyte's native typed data primitives
(`flyte.File`, `flyte.Dir`) automatically manage blob storage transfers between tasks:
datasets live in object storage and are mounted where needed rather than rsynced to
every node. This applies equally in both horizons — the shared object store is the
authoritative data layer whether tasks run via connector or via an ephemeral data plane.

### Honest tension

The principal capability SkyPilot gives researchers that a container-centric
orchestrator does not is the **SSH-into-a-machine, iterate without rebuilding Docker**
workflow. Flyte's devbox-style interactive execution narrows this gap, but it remains
container-centric. For researchers who live in SSH workflows, SkyPilot will feel more
natural.

The honest framing of this integration is not "use Flyte instead of SkyPilot" but
**"use SkyPilot for interactive dev and training execution, and use Flyte as the
production pipeline layer that provides the lineage and auditability the research
workflow does not."** In Horizon 2, this extends to: "SkyPilot provisions the
execution environment; Flyte owns everything that happened inside it."

## Open Questions

- **Dynamic cluster registration (Horizon 2 gate)**: What is the right API surface for
  FlyteAdmin to accept a new data plane at runtime without a Helm restart? Should this
  live in the Union operator, in a new FlyteAdmin endpoint, or in a CRD the control
  plane watches? This is the primary engineering question before Horizon 2 is viable.
- **Bootstrap latency target**: What is the maximum tolerable cold-start time for an
  ephemeral data plane before spot GPU workloads would simply prefer a warm standby
  cluster? The answer determines whether a pre-baked image is sufficient or whether
  Union needs a warm-pool strategy.
- **Armada as a complement**: For organizations running multiple permanent BYOC data
  planes, does Armada's multi-cluster scheduling add enough value on top of FlyteAdmin's
  existing `labelClusterMap` routing to justify the operational complexity? Track this
  as the cluster fleet grows.
- **Flyte devbox vs. SkyPilot SSH**: How close can Flyte's devbox-style interactive
  execution get to the SSH-and-iterate experience researchers expect from SkyPilot, and
  where will the remaining gap matter most? Does the Horizon 2 model (SkyPilot
  bootstraps the cluster; researcher SSHs into a pod on it) close this gap entirely?
- **Lineage consistency across horizons**: How is lineage made consistent for runs
  dispatched via connector (Horizon 1) versus runs executed on an ephemeral data plane
  (Horizon 2)? The shared object store is the anchor, but the metadata surfaced in the
  Flyte UI should be indistinguishable between the two models.
- **CUDA image builds**: How are CUDA-heavy training images built and cached to minimize
  iteration time, whether the build is driven by Flyte, SkyPilot, or a shared image
  registry? Does the pre-baked VM image for Horizon 2 ship with a local image cache, or
  does it pull from a registry on first boot?

## Flyte vs. Dagster in Context

The comparison above focuses on technical capabilities in isolation. In practice, teams
choosing between Flyte and Dagster should also consider:

### Organizational Factors

- **Team expertise**: If your team already uses Dagster for data engineering,
  adopting it for ML may reduce cognitive overhead. If you have Kubernetes experience
  but not Dagster, Flyte may be more familiar.

- **Tool consolidation goals**: The original spec mentions avoiding separate orchestrators
  for research and production. Flyte's unified approach (same system for ETL and training)
  can simplify this; Dagster requires careful architecture to achieve the same.

- **Operational maturity**: Flyte is designed from the start for K8s orchestration,
  while Dagster's K8s support came later. This shows in the depth of integration.

### Hybrid Approaches

Some organizations use a hybrid approach:

1. **Dagster for warehouse-centric ETL** — SQL/dbt/Spark pipelines with partition
   backfills and asset catalog freshness.
2. **Flyte for ML workflows** — Distributed training, model serving, and tracking.
3. **Shared storage layer** — Datasets flow between systems via object storage URIs.

This is workable, but it introduces two operational runbooks and breaks lineage at the
ETL→ML handoff. The preferred architecture for this stack is a single Flyte graph
that covers both data engineering and the full model factory — ETL tasks run in the
home cluster pool while GPU stages target SkyPilot. See `COMPARISON.md` for the
detailed Flyte vs. Dagster analysis.

### Future Directions

Both projects are evolving rapidly:
- **Flyte**: Increasing focus on developer experience (devbox, local execution),
  multi-cluster scheduling, and deeper ML framework integration.
- **Dagster**: Improving distributed compute support, cloud-native capabilities,
  and native integrations with ML tools.

The gap between the two is narrowing, but for ML-heavy workloads with distributed
training requirements, Flyte currently offers more first-class capabilities.
