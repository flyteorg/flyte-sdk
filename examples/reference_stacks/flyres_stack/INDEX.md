# FlyRES Stack Examples Index

This document provides a complete reference to all examples in the FlyRES stack.

For a head-to-head comparison of Flyte 2 vs Dagster (with side-by-side workflow examples), see [COMPARISON.md](./COMPARISON.md).

## Table of Contents

1. [Infrastructure](#infrastructure)
2. [Distributed Training](#distributed-training)
3. [Data Management](#data-management)
4. [Experiment Tracking & Versioning](#experiment-tracking--versioning)
5. [Workflow Orchestration](#workflow-orchestration)
6. [ML Lifecycle](#ml-lifecycle)
7. [Additional Capabilities](#additional-capabilities)

---

## Infrastructure

### 01_image_build_strategy.py
**Purpose**: Multi-stage image building for ML workloads

**Key Features**:
- Base image with PyTorch and CUDA (built infrequently)
- Layered experimental images (changes quickly)
- Efficient build caching

**Dagster Alternative**: None - this is a Flyte-specific optimization pattern

**Run With**:
```bash
uv run --prerelease=allow src/01_image_build_strategy.py
```

---

## Distributed Training

### 02_ray_distributed_training.py
**Purpose**: Distributed training using Flyte-Ray plugin

**Key Features**:
- Ray cluster orchestration with head + worker nodes
- Hugging Face model mount integration
- Parallel data loading across workers

**SkyPilot Alternative**: `sky launch --detach` → Flyte tasks with `RayJobConfig`

**Run With**:
```bash
uv run --prerelease=allow src/02_ray_distributed_training.py
```

### 03_pytorch_fsdp_training.py
**Purpose**: FSDP-style distributed training (Megatron alternative)

**Key Features**:
- PyTorch Elastic configuration for multi-node training
- Distributed Data Parallelism (DDP)
- FSDP-ready architecture

**Megatron Alternative**: Use PyTorch's native `Elastic` config instead of Megatron-LM

**Run With**:
```bash
uv run --prerelease=allow src/03_pytorch_fsdp_training.py
```

---

## Data Management

### 05_hf_mounts_data_loading.py
**Purpose**: Efficient data loading using Hugging Face model mounts

**Key Features**:
- Shared storage for large models and datasets
- Automatic caching with `HF_CACHE_DIR`
- Streaming dataset support
- Optional HF token secret for private models

**Secret Required**: None (HF_TOKEN optional, configured via Flyte secrets if needed)
```bash
flyte secret create hf token=your_token --group data-loading
```

**SkyPilot Alternative**: Volume mounts → HF mounts (specialized for ML)

**Run With**:
```bash
uv run --prerelease=allow src/05_hf_mounts_data_loading.py
```

### 11_data_engineering.py
**Purpose**: ETL pipeline for multimodal ML datasets

**Key Features**:
- Multi-modal data preprocessing (audio, video, text)
- Filtering and curation
- Feature extraction and splits

**Dagster Alternative**: Assets → Flyte tasks with typed data

**Run With**:
```bash
uv run --prerelease=allow src/11_data_engineering.py
```

---

## Experiment Tracking & Versioning

### 04_experiment_tracking.py
**Purpose**: W&B experiment tracking integration

**Key Features**:
- Run metadata and metrics logging
- Model artifact versioning
- Multiple experiment comparison

**Secret Required**: `wandb/api-key` (configured via Flyte secrets)
```bash
flyte secret create wandb api-key=your_wandb_api_key --group experiment-tracking
```

**Dagster Alternative**: Sensors + ops → Flyte tasks with `wandb` integration

**Run With**:
```bash
export WANDB_API_KEY=your_api_key  # or configure as secret
uv run --prerelease=allow src/04_experiment_tracking.py
```

### 09_model_registry_integration.py
**Purpose**: Model versioning and lineage tracking

**Key Features**:
- Versioned model storage
- Comparison between versions
- Full audit trail

**Dagster Alternative**: Manual artifacts → Flyte's typed data + registry integration

**Run With**:
```bash
uv run --prerelease=allow src/09_model_registry_integration.py
```

---

## Workflow Orchestration

### 06_workflow_triggers.py
**Purpose**: Cron-based workflow triggers (Dagster schedule alternative)

**Key Features**:
- Hourly, daily, weekly triggers
- Custom cron schedules with inputs
- Manual triggers

**Dagster Alternative**: `@schedule` decorators → `flyte.Trigger.cron()`

**Run With**:
```bash
uv run --prerelease=allow src/06_workflow_triggers.py
```

### 07_webhook_invocation.py
**Purpose**: Webhook app for external workflow invocation

**Key Features**:
- HTTP webhook triggers
- Event routing (model_ready, data_update, training_complete)
- Async execution model

**SkyPilot Alternative**: `sky launch` + async callbacks → Flyte webhook app env

**Run With**:
```bash
uv run --prerelease=allow src/07_webhook_invocation.py
```

---

## ML Lifecycle

### 08_end_to_end_ml_pipeline.py
**Purpose**: Complete end-to-end ML pipeline

**Key Features**:
- Data ingestion → training → evaluation → deployment
- Full orchestration with Flyte
- Error handling and retries

**Dagster/SkyPilot Combined**: Multiple tools → Single Flyte workflow

**Run With**:
```bash
uv run --prerelease=allow src/08_end_to_end_ml_pipeline.py
```

---

## Additional Capabilities

### 10_serve_models.py
**Purpose**: Model serving (online and batch inference)

**Key Features**:
- Online endpoints with replicas
- Health checks and monitoring
- Batch inference jobs

**SkyPilot Alternative**: `sky serve` → Flyte online endpoint tasks

**Run With**:
```bash
uv run --prerelease=allow src/10_serve_models.py
```

---

## Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Quick overview and comparison matrix |
| `SUMMARY.md` | High-level architecture patterns |
| `QUICKSTART.md` | Getting started guide |
| `examples.yaml` | Detailed example reference |
| `pyproject.toml` | UV project configuration |

---

## Quick Reference

### Most Common Use Cases

1. **Efficient image building**: `01_image_build_strategy.py`
2. **Distributed training with Ray**: `02_ray_distributed_training.py`
3. **Complete ML pipeline**: `08_end_to_end_ml_pipeline.py`
4. **Scheduled jobs**: `06_workflow_triggers.py`
5. **Experiment tracking**: `04_experiment_tracking.py`

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `HF_MOUNT_PATH` | Path to Hugging Face model mounts |
| `HF_TOKEN` | (Optional) Hugging Face authentication token for private models |
| `WANDB_API_KEY` | Weights & Biases API key |
| `WANDB_PROJECT` | W&B project name |

### Flyte Secrets (Recommended for Production)
Configure secrets via Flyte backend:
```bash
flyte secret create wandb api-key=your_wandb_api_key --group experiment-tracking
flyte secret create hf token=your_hf_token --group data-loading
```

This approach is more secure than environment variables and integrates with Flyte's RBAC.
See each example for its specific secrets configuration.

### Running Examples

```bash
# Run a specific example (scripts live in src/)
uv run --prerelease=allow src/<example>.py

# With environment variables
HF_MOUNT_PATH=/models WANDB_API_KEY=key uv run --prerelease=allow src/02_ray_distributed_training.py
```
