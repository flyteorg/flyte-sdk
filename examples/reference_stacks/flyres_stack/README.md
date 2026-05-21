# Flyte + Ray + HF Mounts Reference Stack Examples

This directory contains examples demonstrating Flyte's capabilities for ML workflows,
showing how Flyte can fulfill use cases typically associated with Dagster and other tools.

## Examples Overview

| Example | Description |
|---------|-------------|
| `01_image_build_strategy.py` | Shows multi-stage image building: base CUDA image + quick-turn experimental layer |
| `02_ray_distributed_training.py` | Distributed training using Flyte-Ray plugin with Hugging Face model mounts |
| `03_pytorch_fsdp_training.py` | FSDP distributed training using PyTorch plugin (Megatron alternative) |
| `04_experiment_tracking.py` | W&B experiment tracking integrated into workflow |
| `05_hf_mounts_data_loading.py` | Using Hugging Face model mounts for efficient data loading |
| `06_workflow_triggers.py` | Cron-based triggers as an alternative to Dagster's schedule triggers |
| `07_webhook_invocation.py` | Flyte webhook app for external workflow invocation (SkyPilot alternative) |
| `08_end_to_end_ml_pipeline.py` | Complete ML lifecycle: data prep → training → eval → serving |
| `09_model_registry_integration.py` | Model versioning and lineage tracking |
| `10_serve_models.py` | Online and batch model serving |
| `11_data_engineering.py` | ETL pipeline for multimodal ML datasets (Dagster alternative) |

## Tool Comparisons

For a detailed comparison of Flyte 2 vs. Dagster, see:
- **SPEC.md** - Section "Flyte vs. Dagster: Unbiased Comparison" with capability matrix
- **examples.yaml** - Tool equivalence mappings for each feature

Key differences:
- **Scheduling**: `@schedule` decorators → `flyte.Trigger.cron()`
- **Data handling**: Assets + I/O managers → Typed data primitives
- **Distributed training**: Custom executors → Native Ray/PyTorch plugins

## Prerequisites

- Flyte backend with Ray plugin support
- (Optional) Hugging Face token configured for private models
- (Optional) W&B API key configured for experiment tracking

### Secret Configuration

For examples that use sensitive data (API keys, tokens), configure Flyte secrets:

```bash
# W&B API key for experiment tracking
flyte secret create wandb api-key=your_wandb_api_key

# Hugging Face token for private models
flyte secret create hf token=your_hf_token
```

Or configure them via the Flyte UI/CLI using the secret group names shown in each example.

## Usage

Each example can be run with UV:

```bash
uv run --prerelease=allow examples/reference_stacks/flyres_stack/<example>.py
```

## Comparison to Dagster/SkyPilot

| Feature | Dagster | SkyPilot | Flyte Equivalent |
|---------|---------|----------|------------------|
| Workflow orchestration | Solid/Job graphs | Job scripts | Task workflows |
| Schedule triggers | `@schedule` decorators | Cron on cluster | `flyte.Trigger.cron()` |
| Remote execution | SSH to machine | `sky launch` | Flyte tasks with plugin configs |
| Distributed training | Custom setup | Ray clusters | `RayJobConfig`, `Elastic` config |
| Data mounting | File systems | Volume mounts | HF mounts, Flyte data primitives |
| Experiment tracking | Sensors + ops | CLI logging | W&B integration in tasks |
| ETL/data engineering | Assets + sensors | Custom scripts | Task workflows with typed data |
| Model versioning | Manual artifacts | None | Registry integration |
| Model serving | Custom endpoints | Services | Online endpoints with Flyte |
