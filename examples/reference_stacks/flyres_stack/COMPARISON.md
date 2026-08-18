# Flyte 2 vs Dagster: Head-to-Head Comparison

This report compares **[Flyte 2](https://www.union.ai/docs/v2/union/user-guide/)** (Union’s orchestration layer for AI/ML on Kubernetes) and **[Dagster](https://docs.dagster.io/)** (a data orchestrator with strong asset-centric lineage). It is written for teams evaluating a **single orchestrator** for both data engineering and ML lifecycles—the scenario described in [SPEC.md](./SPEC.md) (FLYRES: Flyte + Ray + optional SkyPilot, Hugging Face, W&B).

**Scope:** Representative, concise workflow examples—not production deployments. Flyte snippets mirror patterns in the [`src/`](./src/) directory (`11_data_engineering.py`, `03_pytorch_fsdp_training.py`, `08_end_to_end_ml_pipeline.py`, etc.). Dagster snippets follow current [asset](https://docs.dagster.io/concepts/assets)-first patterns from the [Dagster docs](https://docs.dagster.io/).

**Last updated:** Flyte 2.x / Union user guide; Dagster 1.13+.

---

## 1. Mental models

| Dimension | Flyte 2 | Dagster |
|-----------|---------|---------|
| **Primary unit** | **Task** in a `TaskEnvironment`; DAG from function dependencies and `depends_on` | **Asset** (data + metadata) or **op** in a **job** / **Definitions** |
| **Orchestration** | Durable **runs** and **actions**; replay from last successful task | Asset materializations; job runs via executor |
| **Data contracts** | Typed task I/O (`flyte.File`, `flyte.Dir`, dataclasses); blob store handled by platform | **I/O managers** + asset metadata; you wire storage explicitly |
| **ML compute** | First-class **task plugins** (Ray, PyTorch Elastic, etc.) and per-task **GPU** `Resources` | Distributed training via **external** runtimes (Ray, K8s jobs) launched from assets/ops |
| **Automation** | **Triggers** on tasks (`Trigger.cron`, `hourly`, `daily`, `Manual`) | **Schedules**, **sensors**, [Declarative Automation](https://docs.dagster.io/guides/automate/declarative-automation) |
| **External events** | [Webhook app / invoke webhook](https://www.union.ai/docs/v2/union/user-guide/task-deployment/invoke-webhook/) | [Sensors](https://docs.dagster.io/guides/automate/sensors), [Labs: Webhooks](https://docs.dagster.io/labs/webhooks) |

**FLYRES takeaway:** Dagster excels when the team thinks in **tables and assets** (catalog, freshness, partitions). Flyte excels when the team thinks in **pipelines of compute steps** with **large artifacts**, **GPUs**, and **reproducible reruns** across prep → train → eval → deploy.

---

## 2. Mapping FLYRES requirements

| SPEC requirement | Flyte 2 | Dagster |
|------------------|---------|---------|
| Multimodal DE ETL | Tasks + parallel `asyncio.gather` / fanout; typed outputs | Assets with `deps=`; parallel ops in jobs |
| Distributed training (FSDP / “Megatron-like”) | [`flyteplugins.pytorch`](./src/03_pytorch_fsdp_training.py), [Ray plugin](./src/02_ray_distributed_training.py) | Custom op/asset calling Ray/K8s; no built-in Elastic/FSDP plugin |
| Data gravity (TB checkpoints, datasets) | [`flyte.File` / `flyte.Dir`](https://www.union.ai/docs/v2/union/user-guide/build-tasks/files-and-directories/); HF mount env | I/O manager + cloud paths; HF via `huggingface_hub` in asset body |
| Experiment tracking | W&B in tasks ([`04_experiment_tracking.py`](./src/04_experiment_tracking.py)) | W&B `Resource` or side-effect in asset |
| Lineage (weights ↔ training ↔ serving) | Run graph + typed artifacts + versioning | [Asset catalog](https://docs.dagster.io/guides/observe/asset-catalog) / lineage metadata |
| Scheduling | [Triggers](https://www.union.ai/docs/v2/union/user-guide/configure-tasks/triggers/) | [Schedules](https://docs.dagster.io/guides/automate/schedules) |
| Event-driven runs | [Invoke webhook](https://www.union.ai/docs/v2/union/user-guide/task-deployment/invoke-webhook/) | [Sensors](https://docs.dagster.io/guides/automate/sensors) / webhooks (labs) |
| Multi-cluster GPU (SkyPilot-style) | Per-task `Resources`; optional SkyPilot subprocess (prospective in SPEC) | Executor + external launcher; not first-class |
| CUDA image iteration | Layered [`Image`](./src/01_image_build_strategy.py) / [container images](https://www.union.ai/docs/v2/union/user-guide/configure-tasks/container-images/) | Docker / PEX; build story is bring-your-own |

---

## 3. Data engineering workflows

### 3.1 Multimodal curation (FLYRES-style)

**Goal:** Ingest raw media → preprocess modalities in parallel → curate → write Parquet for training.

<table>

<tr>
<td>Flyte 2</td>
<td>Dagster</td>
</tr>

<tr>

<td>

Tasks compose by passing structured outputs; parallel modality steps are separate tasks (see [`11_data_engineering.py`](./src/11_data_engineering.py)).

```python
import asyncio
import flyte
from flyte import Image, Resources

env = flyte.TaskEnvironment(
    name="data_eng",
    image=Image.from_debian_base(
        python_version=(3, 12),
    ).with_pip_packages("pandas", "pyarrow"),
)

@env.task
async def ingest_raw_data() -> dict:
    return {
        "records_ingested": 1_000_000,
        "data_types": ["audio", "video", "text_metadata"],
    }

@env.task
async def preprocess_audio(raw_data: dict) -> dict:
    return {
        **raw_data,
        "modality": "audio",
        "processed_records": 950_000,
    }

@env.task
async def preprocess_video(raw_data: dict) -> dict:
    return {
        **raw_data,
        "modality": "video",
        "processed_records": 980_000,
    }

@env.task
async def curate_dataset(
    audio_data: dict,
    video_data: dict,
) -> dict:
    return {
        "curated_records": 850_000,
        "audio": audio_data,
        "video": video_data,
    }

@env.task
async def main() -> dict:
    raw = await ingest_raw_data()
    audio, video = await asyncio.gather(
        preprocess_audio(raw),
        preprocess_video(raw),
    )
    return await curate_dataset(audio, video)

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
```


**Docs:** [Tasks](https://www.union.ai/docs/v2/union/user-guide/core-concepts/tasks/), [Files and directories](https://www.union.ai/docs/v2/union/user-guide/build-tasks/files-and-directories/), [Fanout / parallel execution](https://www.union.ai/docs/v2/union/user-guide/build-tasks/controlling-parallel-execution/).


</td>
<td>

Assets declare dependencies explicitly; the UI shows an asset graph.

```python
import dagster as dg

# Image/resources in dagster.yaml

@dg.asset
def ingest_raw_data() -> dict:
    return {
        "records_ingested": 1_000_000,
        "data_types": ["audio", "video", "text_metadata"],
    }

@dg.asset
def preprocess_audio(ingest_raw_data: dict) -> dict:
    return {
        **ingest_raw_data,
        "modality": "audio",
        "processed_records": 950_000,
    }

@dg.asset
def preprocess_video(ingest_raw_data: dict) -> dict:
    return {
        **ingest_raw_data,
        "modality": "video",
        "processed_records": 980_000,
    }

@dg.asset
def curate_dataset(
    preprocess_audio: dict,
    preprocess_video: dict,
) -> dict:
    return {
        "curated_records": 850_000,
        "audio": preprocess_audio,
        "video": preprocess_video,
    }

defs = dg.Definitions(
    assets=[
        ingest_raw_data,
        preprocess_audio,
        preprocess_video,
        curate_dataset,
    ],
)
```

**Docs:** [Assets](https://docs.dagster.io/concepts/assets), [Asset dependencies](https://docs.dagster.io/tutorial/asset-dependencies).

</td>
</tr>

</table>


### 3.2 Common batch ETL (S3 → transform → warehouse)

**Goal:** Daily partition of files in object storage, normalize, load to warehouse table.

<table>

<tr>
<td>Flyte 2</td>
<td>Dagster</td>
</tr>

<tr>

<td>

Use `flyte.File` / `flyte.Dir` so the platform stages blobs between tasks; optional [caching](https://www.union.ai/docs/v2/union/user-guide/configure-tasks/caching/) skips rework when inputs are unchanged.

```python
import flyte
from flyte import Dir, File, Image

env = flyte.TaskEnvironment(
    name="etl",
    image=Image.from_debian_base(
        python_version=(3, 12),
    ),
)

@env.task(cache="auto")
async def extract(partition_date: str) -> Dir:
    return Dir(path=f"s3://lake/raw/dt={partition_date}")

@env.task
async def transform(raw: Dir) -> File:
    return File(path="s3://lake/staging/cleaned.parquet")

@env.task(
    triggers=flyte.Trigger.cron(
        "0 6 * * *",
        timezone="UTC",
    ),
)
async def load_to_warehouse(
    cleaned: File,
    partition_date: str,
) -> str:
    return f"loaded:{partition_date}"

@env.task
async def main(partition_date: str):
    raw = await extract(partition_date)
    cleaned = await transform(raw)
    return await load_to_warehouse(
        cleaned,
        partition_date,
    )
```

**Docs:** [Caching](https://www.union.ai/docs/v2/union/user-guide/configure-tasks/caching/), [Triggers](https://www.union.ai/docs/v2/union/user-guide/configure-tasks/triggers/).

</td>
<td>

Partitions and I/O managers are the idiomatic pattern.

```python
import dagster as dg
from dagster import DailyPartitionsDefinition

daily = DailyPartitionsDefinition(
    start_date="2024-01-01",
)

@dg.asset(partitions_def=daily)
def extract(
    context: dg.AssetExecutionContext,
) -> str:
    return f"s3://lake/raw/dt={context.partition_key}"

@dg.asset(partitions_def=daily)
def transform(
    context: dg.AssetExecutionContext,
    extract: str,
) -> str:
    return "s3://lake/staging/cleaned.parquet"

@dg.asset(partitions_def=daily)
def load_to_warehouse(
    context: dg.AssetExecutionContext,
    transform: str,
) -> str:
    return f"loaded:{context.partition_key}"

job = dg.define_asset_job("daily_etl", selection="*")

@dg.schedule(cron_schedule="0 6 * * *", job=job)
def daily_etl_schedule(
    context: dg.ScheduleEvaluationContext,
):
    yield dg.RunRequest(
        partition_key=context.scheduled_execution_time.strftime(
            "%Y-%m-%d",
        ),
    )

defs = dg.Definitions(
    assets=[extract, transform, load_to_warehouse],
    schedules=[daily_etl_schedule],
)
```

**Docs:** [Partitions and backfills](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitioned-assets), [Schedules](https://docs.dagster.io/guides/automate/schedules), [I/O managers](https://docs.dagster.io/concepts/io-management/io-managers).

</td>
</tr>

</table>

**Comparison:** Both handle partitioned ETL well. Dagster’s **partition UI and backfill** are mature out of the box. Flyte’s advantage shows when the same ETL **feeds GPU training tasks** in one run graph without a separate “macro orchestrator” launching opaque subprocesses.

---

## 4. ML training workflows

### 4.1 Distributed training with GPUs

**Goal:** Multi-node training (FSDP / DDP-style), pin GPU resources, log to W&B, read base weights from shared HF storage.

<table>

<tr>
<td>Flyte 2</td>
<td>Dagster</td>
</tr>

<tr>

<td>

PyTorch Elastic plugin + per-task resources ([`03_pytorch_fsdp_training.py`](./src/03_pytorch_fsdp_training.py)); Ray alternative in [`02_ray_distributed_training.py`](./src/02_ray_distributed_training.py).

```python
import flyte
import torch
import torch.distributed as dist
import torch.nn as nn
from flyte import Image, Resources
from flyteplugins.pytorch.task import Elastic
from torch.nn.parallel import DistributedDataParallel as DDP

torch_cfg = Elastic(
    nproc_per_node=2,
    nnodes=2,
    max_restarts=3,
)
image = Image.from_debian_base(
    python_version=(3, 12),
).with_pip_packages(
    "flyteplugins-pytorch",
    "torch==2.4.0",
    "wandb",
)

train_env = flyte.TaskEnvironment(
    name="fsdp_train",
    plugin_config=torch_cfg,
    image=image,
    resources=Resources(
        cpu=(4, 8),
        memory=("8Gi", "32Gi"),
        gpu="A10G:1",
    ),
)

def train_loop(epochs: int, lr: float) -> float:
    dist.init_process_group("nccl")
    try:
        rank = dist.get_rank()
        model = nn.Linear(4096, 4096)
        model = DDP(model)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        last_loss = 0.0
        for _ in range(epochs):
            x = torch.randn(32, 4096)
            y = torch.randn(32, 4096)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            last_loss = loss.item()
            if rank == 0:
                print(f"loss={last_loss:.4f}")
        return last_loss
    finally:
        dist.destroy_process_group()

@train_env.task
async def train(hyperparams: dict) -> dict:
    import os
    import wandb

    os.environ["HF_HOME"] = "/models"
    wandb.init(
        project="flyte-experiments",
        config=hyperparams,
    )
    final_loss = train_loop(
        epochs=hyperparams.get("epochs", 3),
        lr=hyperparams.get("lr", 1e-3),
    )
    wandb.log({"train_loss": final_loss})
    return {
        "checkpoint": "s3://models/run-123/ckpt",
        "final_loss": final_loss,
        "run_id": wandb.run.id,
    }
```

**Docs:** [Task plugins](https://www.union.ai/docs/v2/union/user-guide/configure-tasks/task-plugins/), [Resources](https://www.union.ai/docs/v2/union/user-guide/configure-tasks/resources/), [Secrets](https://www.union.ai/docs/v2/union/user-guide/configure-tasks/secrets/) (W&B API key).

</td>
<td>

Training runs inside the asset body (or via Ray); Dagster does not own the process group.

```python
import dagster as dg
import torch
import torch.nn as nn

def train_loop(epochs: int, lr: float) -> float:
    model = nn.Linear(4096, 4096)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    last_loss = 0.0
    for _ in range(epochs):
        x = torch.randn(32, 4096)
        y = torch.randn(32, 4096)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        last_loss = loss.item()
    return last_loss

@dg.asset(deps=["curate_dataset"])
def training_run(
    context: dg.AssetExecutionContext,
    curate_dataset: dict,
) -> dict:
    import wandb

    hyperparams = {"epochs": 3, "lr": 1e-3}
    wandb.init(
        project="dagster-experiments",
        config=hyperparams,
    )
    final_loss = train_loop(
        epochs=hyperparams["epochs"],
        lr=hyperparams["lr"],
    )
    wandb.log({"train_loss": final_loss})
    context.log.info(f"loss={final_loss:.4f}")
    return {
        "checkpoint": "s3://models/run-456/ckpt",
        "final_loss": final_loss,
        "run_id": wandb.run.id,
    }

defs = dg.Definitions(assets=[training_run])
```

**Docs:** [External resources](https://docs.dagster.io/guides/build/external-resources), [Jobs](https://docs.dagster.io/concepts/ops-jobs/jobs-to-be-scheduled).

</td>
</tr>

</table>

**Comparison:** For **FLYRES-style scarce GPUs** and **multi-node PyTorch**, Flyte’s plugin model keeps orchestration and training in one system. Dagster remains a strong **coordinator** if Ray/K8s training already exists, but you maintain two operational surfaces (Dagster run + Ray job).

---

### 4.2 Model weights and data gravity (Hugging Face)

**Goal:** Load large base models from Hugging Face Hub once; reuse cached weights across training tasks without per-node rsync.

<table>

<tr>
<td>Flyte 2</td>
<td>Dagster</td>
</tr>

<tr>

<td>

Upcoming first-class [`Volume`](https://github.com/flyteorg/flyte-sdk/pull/1065) (remote-backed FS with lineage via index `File`); see also [`05_hf_mounts_data_loading.py`](./src/05_hf_mounts_data_loading.py).

```python
import flyte
from flyte.extras import (
    Volume,
    volume_image,
    volume_pod_template,
)

MODEL = "bert-base-uncased"

env = flyte.TaskEnvironment(
    name="hf_vol",
    pod_template=volume_pod_template(
        cache_size_gb=50,
    ),
    image=volume_image(
        flyte.Image.from_debian_base(
            python_version=(3, 12),
        ).with_pip_packages("transformers"),
    ),
)

@env.task
async def cache_base_model() -> Volume:
    vol = Volume.empty(name="hf-weights")
    await vol.mount()
    from transformers import AutoModel

    model = AutoModel.from_pretrained(MODEL)
    model.save_pretrained("/workspace")
    return await vol.commit()

@env.task
async def load_model(vol: Volume) -> dict:
    await vol.mount()
    from transformers import AutoModel

    model = AutoModel.from_pretrained("/workspace")
    n = sum(p.numel() for p in model.parameters())
    return {"parameters": n}
```

**Docs:** [PR #1065 — Volume type](https://github.com/flyteorg/flyte-sdk/pull/1065), [Files and directories](https://www.union.ai/docs/v2/union/user-guide/build-tasks/files-and-directories/).

</td>
<td>

Same libraries; weights live on a shared path string downstream assets consume.

```python
import dagster as dg

MODEL = "bert-base-uncased"
SHARED = "/shared/models/bert-base-uncased"

@dg.asset
def cache_base_model(
    context: dg.AssetExecutionContext,
) -> str:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(MODEL)
    model.save_pretrained(SHARED)
    context.log.info(f"Cached weights at {SHARED}")
    return SHARED

@dg.asset
def load_model(
    context: dg.AssetExecutionContext,
    cache_base_model: str,
) -> dict:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(cache_base_model)
    n = sum(p.numel() for p in model.parameters())
    context.log.info(f"parameters={n}")
    return {"parameters": n}
```

**Docs:** [I/O managers](https://docs.dagster.io/concepts/io-management/io-managers).

</td>
</tr>

</table>

**Comparison:** Flyte `Volume` ties **metadata index + blob chunks** into the run graph (cache, fork, replay). Dagster achieves similar outcomes with **shared volumes + path assets**, but lineage across TB-scale weight trees is more manual (SPEC § data gravity).

---

### 4.3 Layered CUDA images (slow base + fast experimental layer)

**Goal:** Build CUDA/PyTorch base images infrequently; layer fast-changing experimental packages on top.

<table>

<tr>
<td>Flyte 2</td>
<td>Dagster</td>
</tr>

<tr>

<td>

[`01_image_build_strategy.py`](./src/01_image_build_strategy.py).

```python
import flyte
from flyte import Image

base = (
    Image.from_debian_base(
        name="ml-base",
        python_version=(3, 12),
    )
    .with_pip_packages("torch==2.4.0")
    .with_apt_packages("cuda-compiler-12-4")
)
experimental = base.with_pip_packages(
    "transformers==4.42.0",
    "peft==0.11.0",
)

base_env = flyte.TaskEnvironment(
    name="base",
    image=base,
)
exp_env = flyte.TaskEnvironment(
    name="experimental",
    image=experimental,
)

@base_env.task
async def pretrain() -> str: ...

@exp_env.task
async def fine_tune(checkpoint: str) -> str: ...
```

**Docs:** [Container images](https://www.union.ai/docs/v2/union/user-guide/configure-tasks/container-images/).

</td>
<td>

Images are not a core Dagster concept; execution environment is configured on the **executor** (Docker, K8s, PEX).

```python
# dagster.yaml (conceptual)
# run_launcher:
#   module: dagster_k8s
#   class: K8sRunLauncher
#   config:
#     image_pull_policy: IfNotPresent
#     job_image: myregistry/ml-base:cuda12.4
#
# Experimental tag bumped in CI:
# job_image: myregistry/ml-experimental:tf-4.42
```

**Docs:** [Deployment](https://docs.dagster.io/deployment).

</td>
</tr>

</table>

**Comparison:** Flyte treats **image as code** next to tasks; Dagster treats **environment as deployment config**—better for classic DE, less tailored to hour-long CUDA rebuild cycles.

---

## 5. ML evaluation and benchmarking

**Goal:** After training, run benchmark suite, gate promotion on metrics, register artifact with lineage to dataset + training run.

<table>

<tr>
<td>Flyte 2</td>
<td>Dagster</td>
</tr>

<tr>

<td>

From [`08_end_to_end_ml_pipeline.py`](./src/08_end_to_end_ml_pipeline.py)—eval and registry are tasks in the same graph as training.

```python
import flyte

base_env = flyte.TaskEnvironment(name="ml_eval")

@base_env.task
async def evaluate_model(
    training_result: dict,
) -> dict:
    metrics = training_result["final_metrics"]
    return {
        **training_result,
        "accuracy": metrics.get("accuracy", 0.82),
        "f1_score": 0.80,
        "evaluation_status": "success",
    }

@base_env.task
async def register_model(
    evaluation: dict,
) -> dict:
    acc = evaluation["accuracy"]
    return {
        **evaluation,
        "model_registry": {
            "name": "text-classifier-v1",
            "version": "1.0.0",
        },
        "deployment_ready": acc > 0.7,
    }

@base_env.task
async def main(training_result: dict):
    evaluation = await evaluate_model(
        training_result,
    )
    return await register_model(evaluation)
```

**Docs:** [Runs and actions](https://www.union.ai/docs/v2/union/user-guide/core-concepts/runs-and-actions/), [Links / reports](https://www.union.ai/docs/v2/union/user-guide/build-tasks/links/).

</td>
<td>

[Asset checks](https://docs.dagster.io/guides/test/asset-checks) fit evaluation gates.

```python
import dagster as dg

@dg.asset(deps=[training_run])
def evaluation_metrics(
    context: dg.AssetExecutionContext,
    training_run: dict,
) -> dict:
    return {"accuracy": 0.82, "f1_score": 0.80}

@dg.asset_check(asset=evaluation_metrics)
def accuracy_threshold(
    evaluation_metrics: dict,
) -> dg.AssetCheckResult:
    acc = evaluation_metrics["accuracy"]
    return dg.AssetCheckResult(
        passed=acc > 0.7,
        metadata={"accuracy": acc},
    )

@dg.asset(deps=[evaluation_metrics])
def registered_model(
    context: dg.AssetExecutionContext,
    evaluation_metrics: dict,
) -> str:
    return "registry://text-classifier-v1/1.0.0"

defs = dg.Definitions(
    assets=[evaluation_metrics, registered_model],
    asset_checks=[accuracy_threshold],
)
```

**Docs:** [Asset checks](https://docs.dagster.io/guides/test/asset-checks), [Asset health](https://docs.dagster.io/guides/observe/asset-health-status).

</td>
</tr>

</table>

**Comparison:** Dagster’s **asset checks** are excellent for **data quality and metric gates** in a catalog mental model. Flyte emphasizes **end-to-end run replay**: the same run ID ties together training output URI, eval task version, and deploy task—useful when leadership asks “what changed between model weights and serving runtime?” (SPEC § reproducibility).

---

## 6. Automation: schedules, events, and external triggers

**Goal:** Run pipelines on a cron, react to external events (S3 drops, CI callbacks), and support manual backfills.

| Pattern | Flyte 2 | Dagster |
|---------|---------|---------|
| Cron | `flyte.Trigger.cron(...)` on task ([`06_workflow_triggers.py`](./src/06_workflow_triggers.py)) | `@dg.schedule` + `RunRequest` |
| Manual / backfill | `flyte.Trigger.Manual()` with inputs | Launchpad / backfill partition APIs |
| File / queue event | Flyte app polling object storage (§6.3) | `@dg.sensor` polling S3/SQS (§6.3) |
| HTTP callback | [Flyte webhook app](https://www.union.ai/docs/v2/union/user-guide/native-app-integrations/flyte-webhook/) (§6.2) | [REST / GraphQL API](https://docs.dagster.io/api/rest-api) `launchRun` (§6.2) |

### 6.1 Cron schedule

<table>

<tr>
<td>Flyte 2</td>
<td>Dagster</td>
</tr>

<tr>

<td>

```python
import flyte
from datetime import datetime

env = flyte.TaskEnvironment(name="scheduled")

@env.task(
    triggers=flyte.Trigger.cron(
        "0 6 * * *",
        timezone="UTC",
    ),
)
async def daily_etl(
    trigger_time: datetime,
    partition_date: str,
) -> str:
    ts = trigger_time.isoformat()
    return f"loaded:{partition_date} at {ts}"
```

**Docs:** [Triggers](https://www.union.ai/docs/v2/union/user-guide/configure-tasks/triggers/).

</td>
<td>

```python
import dagster as dg

@dg.schedule(cron_schedule="0 6 * * *", job=daily_etl_job)
def daily_etl_schedule(
    context: dg.ScheduleEvaluationContext,
):
    yield dg.RunRequest(
        run_key=context.scheduled_execution_time.isoformat(),
        run_config={
            "ops": {
                "daily_etl": {
                    "config": {
                        "partition_date": "2024-01-01",
                    },
                },
            },
        },
    )
```

**Docs:** [Schedules](https://docs.dagster.io/guides/automate/schedules).

</td>
</tr>

</table>

### 6.2 HTTP webhook (external callback → run)

<table>

<tr>
<td>Flyte 2</td>
<td>Dagster</td>
</tr>

<tr>

<td>

Deploy a [`FlyteWebhookAppEnvironment`](https://www.union.ai/docs/v2/union/user-guide/native-app-integrations/flyte-webhook/) app; callers POST to run a task ([`run_webhook_env.py`](../../apps/run_webhook_env.py)).

```python
import flyte
from flyte.app.extras import FlyteWebhookAppEnvironment

task_env = flyte.TaskEnvironment(
    name="on_event",
    image=flyte.Image.from_debian_base(),
)

@task_env.task
async def on_external_event(
    payload: dict,
) -> dict:
    return {"status": "processed", **payload}

webhook_env = FlyteWebhookAppEnvironment(
    name="flyte-webhook",
    endpoint_groups=["core", "task", "run"],
    task_allowlist=["on_event.on_external_event"],
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=True,
)

# flyte.deploy(task_env)
# served = flyte.serve(webhook_env)
# served.activate()
# POST .../run-task/{domain}/{project}/on_event.on_external_event
#      {"payload": {"batch_id": "batch-42"}}
```

**Docs:** [Flyte webhook app](https://www.union.ai/docs/v2/union/user-guide/native-app-integrations/flyte-webhook/), [Invoke webhook](https://www.union.ai/docs/v2/union/user-guide/task-deployment/invoke-webhook/).

</td>
<td>

Dagster has **outgoing** [webhook alerts](https://docs.dagster.io/guides/labs/webhook-alerts); to **start a run from HTTP**, use the [REST / GraphQL API](https://docs.dagster.io/api/rest-api) (`launchRun`).

```python
import httpx

DAGSTER_URL = "https://dagster.example.com"
LAUNCH_RUN = """
mutation LaunchRun($jobName: String!, $runConfig: RunConfigData!) {
  launchRun(executionParams: {
    selector: { repositoryLocationName: "repo", jobName: $jobName }
    runConfigData: $runConfig
  }) { __typename ... on LaunchRunSuccess { run { runId } } }
}
"""

def trigger_via_webhook(
    job_name: str,
    payload: dict,
) -> str:
    variables = {
        "jobName": job_name,
        "runConfig": {
            "ops": {
                "on_external_event": {
                    "config": payload,
                },
            },
        },
    }
    resp = httpx.post(
        f"{DAGSTER_URL}/graphql",
        json={"query": LAUNCH_RUN, "variables": variables},
        headers={"Dagster-Cloud-Api-Token": "..."},
    )
    resp.raise_for_status()
    return resp.json()["data"]["launchRun"]["run"]["runId"]

# CI/CD: POST-equivalent call after build completes
trigger_via_webhook(
    "training_job",
    {"batch_id": "batch-42"},
)
```

**Docs:** [REST API](https://docs.dagster.io/api/rest-api), [GraphQL API](https://docs.dagster.io/api/graphql), [Webhook alerts (outgoing)](https://docs.dagster.io/guides/labs/webhook-alerts).

</td>
</tr>

</table>

### 6.3 File sensor (poll object storage → run)

<table>

<tr>
<td>Flyte 2</td>
<td>Dagster</td>
</tr>

<tr>

<td>

A long-running **Flyte app** polls a prefix and starts a task when new files land ([`07_webhook_invocation.py`](./src/07_webhook_invocation.py) for task logic).

```python
import asyncio
import flyte
from fastapi import FastAPI
from flyte.app.extras import (
    FastAPIAppEnvironment,
    FastAPIPassthroughAuthMiddleware,
)

task_env = flyte.TaskEnvironment(name="ingest")

@task_env.task
async def ingest_new_files(
    prefix: str,
    batch_id: str,
) -> str:
    return f"ingested:{prefix}:{batch_id}"

app = FastAPI()
app.add_middleware(
    FastAPIPassthroughAuthMiddleware,
    excluded_paths={"/health"},
)

@app.on_event("startup")
async def poll_s3():
    seen: set[str] = set()
    while True:
        for key in list_new_keys("s3://bucket/incoming/"):
            if key not in seen:
                seen.add(key)
                await flyte.run(
                    ingest_new_files,
                    prefix=key,
                    batch_id=key,
                )
        await asyncio.sleep(60)

sensor_app = FastAPIAppEnvironment(
    name="s3-file-sensor",
    app=app,
    scaling=flyte.app.Scaling(replicas=1),
)

# flyte.deploy(task_env); flyte.serve(sensor_app).activate()
```

**Docs:** [FastAPI app](https://www.union.ai/docs/v2/union/user-guide/native-app-integrations/fastapi-app/), [Apps](https://www.union.ai/docs/v2/union/user-guide/core-concepts/apps/).

</td>
<td>

Native **`@dg.sensor`** polls storage and yields `RunRequest`s.

```python
import dagster as dg

@dg.asset
def ingest_new_files(
    context: dg.AssetExecutionContext,
) -> str:
    cfg = context.op_config
    prefix = cfg["prefix"]
    batch_id = cfg["batch_id"]
    context.log.info(f"ingest {prefix} {batch_id}")
    return f"ingested:{prefix}:{batch_id}"

ingest_job = dg.define_asset_job(
    "ingest_job",
    selection="ingest_new_files",
)

@dg.sensor(job=ingest_job, minimum_interval_seconds=60)
def s3_file_sensor(
    context: dg.SensorEvaluationContext,
):
    for key in list_new_keys("s3://bucket/incoming/"):
        yield dg.RunRequest(
            run_key=key,
            run_config={
                "ops": {
                    "ingest_new_files": {
                        "config": {
                            "prefix": key,
                            "batch_id": key,
                        },
                    },
                },
            },
        )

defs = dg.Definitions(
    assets=[ingest_new_files],
    jobs=[ingest_job],
    sensors=[s3_file_sensor],
)
```

**Docs:** [Sensors](https://docs.dagster.io/guides/automate/sensors).

</td>
</tr>

</table>

---

## 7. Lineage, caching, and the “model factory” vision

The [Poolside model factory](https://poolside.ai/blog/introducing-the-model-factory) pattern (orchestrated prep → train → eval → serve) maps to:

| Capability | Flyte 2 | Dagster |
|------------|---------|---------|
| **Skip redundant work** | Task-level [caching](https://www.union.ai/docs/v2/union/user-guide/configure-tasks/caching/) from input + code hash | Op/asset caching via policies |
| **Artifact typing** | Native `File`/`Dir`/structured types | Asset keys + metadata |
| **Cross-team catalog** | Union/Flyte UI run graph | [Asset catalog](https://docs.dagster.io/guides/observe/asset-catalog) (Dagster+) |
| **GPU pipeline + DE in one tool** | Single task graph | Single Definitions object; training still often external |

**Data gravity (SPEC):** Flyte is designed so **datasets and checkpoints are references**, not repeated rsync payloads. Dagster can achieve the same with disciplined I/O managers and shared volumes, but it is not the default ergonomics for multi-terabyte model files.

**SkyPilot (prospective):** Neither replaces the other; see [SPEC.md § Flyte ↔ SkyPilot](./SPEC.md#flyte--skypilot-integration-prospective). Dagster would typically wrap `sky launch` in an op—same “opaque subprocess” risk Flyte avoids when training stays a first-class plugin task.

---

## 8. Developer experience

| Topic | Flyte 2 | Dagster |
|-------|---------|---------|
| **Local iteration** | [`flyte.with_runcontext(mode="local")`](https://www.union.ai/docs/v2/union/user-guide/run-modes/run-locally/), `/// script` + uv in examples | `dagster dev` — strong local story without K8s |
| **K8s production** | Core path: `flyte.deploy()`, [run on cluster](https://www.union.ai/docs/v2/union/user-guide/run-modes/run-on-a-remote-cluster/) | [Deployment](https://docs.dagster.io/deployment) — K8s via executors, more assembly |
| **Interactive research (SSH)** | Devbox / containers; gap vs SSH-first tools (SPEC) | Local assets; remote SSH not a Dagster feature either |
| **Testing** | [Unit testing tasks](https://www.union.ai/docs/v2/union/user-guide/build-tasks/unit-testing-tasks/) | [Testing assets](https://docs.dagster.io/guides/test/testing-assets) |

---

## 9. When to choose which (for FLYRES-like teams)

### Prefer Flyte 2 when

- One orchestrator must cover **DE ETL and ML** (train, eval, deploy) with **GPU** and **distributed plugins**.
- **Lineage and rerun fidelity** across checkpoints, eval metrics, and serving are compliance- or incident-driven requirements.
- **Large artifacts** and **semantic caching** dominate cost and iteration time.
- You are already on or targeting **Kubernetes** (Union or self-hosted Flyte 2).

### Prefer Dagster when

- The organization is already **asset-centric** for analytics/DE with Dagster+ catalog, checks, and partitions.
- Workloads are primarily **SQL/dbt/Spark-style** pipelines with occasional ML jobs launched externally.
- **Partition backfills and data contracts** matter more than multi-node PyTorch orchestration.

### Hybrid (common in practice)

- Dagster for **warehouse-centric ETL**; Flyte for **ML factory** stages—connected via object storage URIs. This matches SPEC’s “avoid two orchestrators” tension: workable, but operational overhead remains.

---

## 10. Side-by-side cheat sheet

| Task | Flyte 2 entrypoint | Dagster entrypoint |
|------|-------------------|-------------------|
| Define step | `@env.task` | `@dg.asset` / `@dg.op` |
| Wire DAG | Function args + `depends_on` | `deps=[...]` / `define_asset_job` |
| Schedule | `triggers=flyte.Trigger.cron(...)` | `@dg.schedule` |
| External event | Flyte webhook app (§6.2) | GraphQL `launchRun` (§6.2) |
| File sensor | Flyte polling app (§6.3) | `@dg.sensor` (§6.3) |
| Persist dataset | `flyte.Dir` / `flyte.File` | I/O manager + asset |
| GPU | `Resources(gpu="A10G:1")` | Executor / external cluster |
| Distributed train | `Elastic`, `RayJobConfig` plugins | Ray/K8s from asset |
| Metric gate | Task logic or downstream task | `@dg.asset_check` |
| Docs home | [Union user guide](https://www.union.ai/docs/v2/union/user-guide/) | [Dagster docs](https://docs.dagster.io/) |

---

## 11. Further reading

**Flyte / Union**

- [User guide overview](https://www.union.ai/docs/v2/union/user-guide/)
- [Triggers](https://www.union.ai/docs/v2/union/user-guide/configure-tasks/triggers/)
- [Invoke webhook](https://www.union.ai/docs/v2/union/user-guide/task-deployment/invoke-webhook/)
- [Caching](https://www.union.ai/docs/v2/union/user-guide/configure-tasks/caching/)
- [Files and directories](https://www.union.ai/docs/v2/union/user-guide/build-tasks/files-and-directories/)

**Dagster**

- [Overview](https://docs.dagster.io/)
- [Assets tutorial](https://docs.dagster.io/tutorial/tutorial-etl)
- [Schedules](https://docs.dagster.io/guides/automate/schedules)
- [Sensors](https://docs.dagster.io/guides/automate/sensors)
- [Partitions](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitioned-assets)
- [Asset checks](https://docs.dagster.io/guides/test/asset-checks)

**This repo**

- [SPEC.md](./SPEC.md) — requirements and Flyte vs Dagster summary tables
- [README.md](./README.md) / [INDEX.md](./INDEX.md) — runnable Flyte examples
