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
- **Flyte**: Macro orchestrator for both AI research workflows and data engineering
  pipelines; owns lineage, versioning, and tracking across the full lifecycle.
- **SkyPilot**: Multi-cluster Kubernetes management and serving (where Flyte does not
  directly provide that capability). The spec should make explicit where SkyPilot and
  Flyte overlap, where they are complementary, and where one can replace the other.
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

## Flyte ↔ SkyPilot Integration (Prospective)

This section describes a prospective integration pattern for organizations that already
run SkyPilot, or that want SkyPilot's interactive / multi-cluster scheduling story while
adopting Flyte as the pipeline layer. It is presented as a possibility rather than a
prescription — adoption can be incremental, and either side of the integration can be
collapsed into Flyte over time as Flyte's own capabilities (e.g. multi-cluster
scheduling, devbox-style interactive execution) mature.

### Tool boundaries

Flyte and SkyPilot operate at genuinely different abstraction layers and are
complementary rather than competing in this configuration:

- **SkyPilot is a compute control plane.** It manages dev pods, jobs, and services
  across multiple Kubernetes clusters through a single pane of glass — handling GPU
  scheduling, SSH access, and code syncing. A SkyPilot task declares resource
  requirements, data to sync, setup commands, and run commands, and can then be
  launched on any available infra (Kubernetes, Slurm, cloud).
- **Flyte is a workflow orchestration and durable execution platform.** It owns the
  DAG, versioning, data lineage, caching, retries, and the audit trail of what ran,
  with what inputs, and what came out.

### Integration Best Practices

When integrating Flyte with SkyPilot:

1. **Use Flyte for orchestration, SkyPilot for compute**: Let Flyte manage workflow logic,
   retries, and lineage while delegating GPU scheduling to SkyPilot.

2. **Handle async jobs properly**: Use `detach_run=True` in SkyPilot launches and implement
   polling with proper timeouts in Flyte tasks to avoid hanging workflows.

3. **Collect artifacts through shared storage**: Have SkyPilot jobs write outputs to S3/GCS
   (not local disk), then have Flyte read those paths. This ensures data is accessible across
   the entire workflow.

4. **Pass minimal configuration, not code**: Avoid syncing large codebases via SkyPilot's
   `file_mounts`. Instead, build container images with your training code and reference them
   in SkyPilot tasks.

5. **LogSkyPilot job IDs to Flyte lineage**: Include the SkyPilot job ID as a task output so
   you can trace back from Flyte runs to the actual compute jobs.

6. **Use Flyte for data prep, SkyPilot for training**: Preprocessing is typically CPU-bound
   and fits well in Flyte; GPU-intensive training benefits from SkyPilot's cluster management.

### What Flyte adds on top of SkyPilot

Gaps around "what changed between model weights and serving runtime across releases"
are lineage and auditability concerns, not scheduling concerns. SkyPilot does not
address them; Flyte does:

- **Semantic caching.** Flyte knows whether a preprocessing step has already run on
  this exact dataset version and can skip it. SkyPilot does not cache at that semantic
  level.
- **Data lineage.** Every artifact (dataset version, checkpoint, eval score) is typed
  and tracked through the workflow graph. When a model behaves differently in prod, it
  is possible to trace back to exactly which data and which training run produced it.
- **Durable execution.** If a node fails mid-pipeline, Flyte resumes from the last
  successful task. SkyPilot handles job-level retries but not pipeline-level
  resumption.
- **Versioned reproducibility.** Flyte versions task code alongside the execution, so
  any historical pipeline can be re-run exactly.

This combination matches a "model factory" pattern: SkyPilot's API server enables
asynchronous execution for hyperparameter sweeps and experiment batches in parallel,
while Flyte wraps that with provenance, retries, and pipeline-level observability for
production workflows.

### Data gravity

Data gravity is the strongest differentiator for an AI-native orchestrator in this
stack. Long rsync times, hour-long Docker builds for CUDA packages, and large
checkpoints are all data gravity problems. Flyte's native typed data primitives
(`flyte.File`, `flyte.Dir`) automatically manage blob storage transfers between tasks:
datasets live in object storage and are mounted where needed rather than rsynced to
every node. This is qualitatively different from a generic macro orchestrator kicking
off a SkyPilot job as a subprocess and treating its inputs and outputs as opaque.

### Honest tension

The principal capability SkyPilot gives researchers that a container-centric
orchestrator does not is the **SSH-into-a-machine, iterate without rebuilding Docker**
workflow. This is often the deciding factor for research teams when first choosing
between SkyPilot and a pipeline-centric tool. Flyte's devbox-style interactive
execution narrows this gap by letting tasks run interactively on remote infra, but it
remains container-centric. For researchers who live in Jupyter and SSH workflows,
SkyPilot will feel more natural.

The honest framing of this integration is therefore not "use Flyte instead of
SkyPilot" but **"use SkyPilot for interactive dev and training execution, and use
Flyte as the production pipeline layer that provides the lineage and auditability the
research workflow does not."**

## Open Questions

- Under what conditions should Flyte subsume more of SkyPilot's responsibilities
  (multi-cluster scheduling, serving) versus continue delegating to SkyPilot? What is
  the migration path in either direction?
- How close can Flyte's devbox-style interactive execution get to the SSH-and-iterate
  experience that researchers expect from SkyPilot, and where will the remaining gap
  matter most?
- How is tracking and lineage made consistent across training, artifacts, and serving
  runtime so that "what changed between releases" is always answerable, including for
  runs that were dispatched via SkyPilot?
- How are CUDA-heavy training images built and cached to minimize iteration time,
  whether the build is driven by Flyte, SkyPilot, or a shared image registry?

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

Many organizations use a hybrid approach:

1. **Dagster for data engineering** - Orchestrate ETL pipelines that prepare datasets
2. **Flyte for ML workflows** - Handle distributed training, model serving, and tracking
3. **Shared storage layer** - Datasets flow between systems via object storage

This pattern leverages each tool's strengths: Dagster for familiar data engineering
patterns and Flyte for ML-specific capabilities like GPU scheduling and distributed
training plugins.

### Future Directions

Both projects are evolving rapidly:
- **Flyte**: Increasing focus on developer experience (devbox, local execution),
  multi-cluster scheduling, and deeper ML framework integration.
- **Dagster**: Improving distributed compute support, cloud-native capabilities,
  and native integrations with ML tools.

The gap between the two is narrowing, but for ML-heavy workloads with distributed
training requirements, Flyte currently offers more first-class capabilities.
