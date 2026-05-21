# FlyRES Stack Examples Summary

This document provides a high-level overview of the FlyRES stack examples, which demonstrate how Flyte can fulfill ML workflow requirements that are typically addressed by Dagster and SkyPilot.

## Overview

The **FlyRES** stack combines:
- **Flyte**: Macro orchestration for ML workflows
- **Ray**: Distributed compute runtime for training
- **SkyPilot** (optional): Multi-cluster scheduling (replaced by Flyte)
- **HF Mounts**: Efficient model/dataset storage
- **W&B**: Experiment tracking

## Example Categories

### 1. Infrastructure & Build Strategy

| File | Purpose |
|------|---------|
| `01_image_build_strategy.py` | Multi-stage image building with base (PyTorch/CUDA) + experimental layers for faster iteration |

**Key Takeaway**: Flyte's Image API allows building images in layers, caching heavy dependencies and only rebuilding the changing parts.

### 2. Distributed Training

| File | Purpose |
|------|---------|
| `02_ray_distributed_training.py` | Uses Ray plugin for distributed training with model mounts |
| `03_pytorch_fsdp_training.py` | Uses PyTorch Elastic plugin (FSDP-style) as Megatron alternative |

**Key Takeaway**: Flyte plugins handle distributed compute orchestration without manual cluster management.

### 3. Data Management

| File | Purpose |
|------|---------|
| `05_hf_mounts_data_loading.py` | Uses Hugging Face model mounts for efficient data access |

**Key Takeaway**: HF mounts provide shared storage accessible by all nodes, avoiding redundant downloads.

### 4. Experiment Tracking & Versioning

| File | Purpose |
|------|---------|
| `04_experiment_tracking.py` | W&B integration for run tracking and metrics |
| `09_model_registry_integration.py` | Model versioning with full lineage |

**Key Takeaway**: Flyte tasks can integrate with external tools (W&B) while maintaining workflow-level tracking.

### 5. Workflow Orchestration

| File | Purpose |
|------|---------|
| `06_workflow_triggers.py` | Cron-based triggers as alternative to Dagster schedules |
| `07_webhook_invocation.py` | Webhook app for external systems (SkyPilot alternative) |

**Key Takeaway**: Flyte provides first-class trigger support and webhook integration.

### 6. Complete ML Lifecycle

| File | Purpose |
|------|---------|
| `08_end_to_end_ml_pipeline.py` | End-to-end workflow: data → train → eval → deploy → monitor |

**Key Takeaway**: Flyte orchestrates the complete lifecycle with proper error handling and retry logic.

### 7. Additional Capabilities

| File | Purpose |
|------|---------|
| `10_serve_models.py` | Model serving (online + batch inference) |
| `11_data_engineering.py` | ETL pipeline for multimodal datasets |

**Key Takeaway**: Flyte handles both research and data engineering workloads.

## Comparison to Other Tools

### vs. Dagster
- **Schedules**: Use `flyte.Trigger.cron()` instead of `@schedule`
- **External triggers**: Use webhook app env instead of sensors
- **Data tracking**: Flyte's typed data primitives provide better lineage than manual artifacts

### vs. SkyPilot
- **Cluster management**: Use Flyte's plugin configs (Ray, PyTorch Elastic) instead of `sky launch`
- **Multi-cluster**: Flyte handles cluster scheduling natively
- **Interactive dev**: SkyPilot still has SSH advantage, but Flyte devbox is catching up

## Architecture Patterns

### Efficient Image Builds
```python
base = Image.from_debian_base().with_pip_packages("torch")
experimental = base.with_pip_packages("transformers")  # Reuses cached base
```

### Distributed Training with Ray
```python
@task(plugin_config=RayJobConfig(worker_replicas=4))
async def distributed_train(): ...
```

### Cron-Based Triggers
```python
@task(triggers=flyte.Trigger.hourly())
async def scheduled_task(trigger_time: datetime): ...
```

### Webhook Integration
```python
# Configure webhook app env to receive external triggers
@task
async def webhook_handler(payload: dict, metadata: dict): ...
```

## Running the Examples

Each example can be run independently:

```bash
# With default settings
uv run --prerelease=allow examples/reference_stacks/flyres_stack/01_image_build_strategy.py

# With custom configuration
HF_MOUNT_PATH=/custom/path WANDB_API_KEY=abc uv run --prerelease=allow examples/reference_stacks/flyres_stack/04_experiment_tracking.py
```

## Prerequisites

- Flyte backend with Ray plugin support
- (Optional) Hugging Face token for model mounts
- (Optional) W&B API key for experiment tracking
- Sufficient GPU resources for training examples

## Next Steps

1. Review `08_end_to_end_ml_pipeline.py` for a complete workflow overview
2. Explore `01_image_build_strategy.py` to understand image optimization
3. Check `06_workflow_triggers.py` and `07_webhook_invocation.py` for workflow orchestration
4. See `09_model_registry_integration.py` for versioning and lineage
