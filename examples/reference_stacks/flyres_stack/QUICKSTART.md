# FlyRES Stack Examples Quick Start

This guide helps you get started with the FlyRES stack examples.

## Prerequisites

- Python 3.12+
- UV package manager (`pip install uv`)
- Flyte backend configured (local or remote)
- Optional: W&B account for experiment tracking
- Optional: Hugging Face token for model mounts

### Secret Configuration

Configure secrets in your Flyte backend:

```bash
# W&B API key for experiment tracking (examples 04, 08)
flyte secret create wandb api-key=your_wandb_api_key --group experiment-tracking

# Hugging Face token for private models (example 05)
flyte secret create hf token=your_hf_token --group data-loading
```

Or configure via Flyte UI in the Secrets tab.

## Installation

```bash
cd examples/reference_stacks/flyres_stack

# Create a virtual environment with UV
uv sync --python 3.12

# Or run directly with uv run
uv run --prerelease=allow <example>.py
```

## Configuration

Set environment variables as needed:

```bash
export HF_MOUNT_PATH=/path/to/hf/mount
export HF_TOKEN=your_huggingface_token
export WANDB_API_KEY=your_wandb_api_key
export WANDB_PROJECT=flyte-examples
```

## Running Examples

### 1. Image Build Strategy (Standalone)
```bash
uv run --prerelease=allow 01_image_build_strategy.py
```

This demonstrates multi-stage image building without requiring Flyte backend.

### 2. Distributed Training (Requires Flyte Backend)
```bash
# Requires Ray plugin enabled in Flyte config
uv run --prerelease=allow 02_ray_distributed_training.py

# With PyTorch Elastic
uv run --prerelease=allow 03_pytorch_fsdp_training.py
```

### 3. Experiment Tracking (Requires W&B)
```bash
export WANDB_API_KEY=your_api_key
uv run --prerelease=allow 04_experiment_tracking.py
```

### 4. End-to-End ML Pipeline
```bash
uv run --prerelease=allow 08_end_to_end_ml_pipeline.py
```

This runs the complete workflow: data ingestion → training → serving.

## Example Workflow

Here's a typical development workflow:

1. **Start with image building** (`01_image_build_strategy.py`)
   - Understand how to build efficient images with caching

2. **Explore distributed training** (`02_ray_distributed_training.py` or `03_pytorch_fsdp_training.py`)
   - See how Flyte handles Ray clusters and PyTorch elastic training

3. **Run full pipeline** (`08_end_to_end_ml_pipeline.py`)
   - Experience the complete ML lifecycle orchestrated by Flyte

4. **Customize for your use case**
   - Adapt the examples to your specific data, models, and infrastructure

## Troubleshooting

### Ray cluster not connecting
- Ensure network policies allow pod-to-pod communication
- Check that Ray plugin is enabled in Flyte config

### HF mount permissions
- Verify mount path exists and is accessible by all nodes
- Set proper environment variables (`HF_HOME`, `TRANSFORMERS_CACHE`)

### W&B authentication errors
- Verify API key is set correctly
- Check project name and entity settings

## Next Steps

1. Read the detailed documentation in `examples.yaml`
2. Review `SUMMARY.md` for architecture patterns
3. Explore individual examples to understand specific features
4. Integrate with your existing ML infrastructure

## Getting Help

- [Flyte Documentation](https://docs.flyte.org)
- [Ray Plugin Docs](https://docs.flyte.org/projects/flyteplugins/en/latest/ray/)
- [PyTorch Plugin Docs](https://docs.flyte.org/projects/flyteplugins/en/latest/pytorch/)
