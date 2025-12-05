"""
Cache a Hugging Face model as a Flyte directory.

This script downloads a model from Hugging Face Hub and saves it as a flyte.io.Dir
that can be used by the VLLMAppEnvironment.

Prerequisites
-------------

1. Create a Hugging Face API token and store it as a secret:

   ```
   flyte create secret --name HF_TOKEN
   ```

Usage
-----

Run the task to cache the Qwen3-0.6B model:

```
flyte run examples/genai/vllm/cache_model.py cache_model \
    --model_id Qwen/Qwen3-0.6B
```

Or with a custom output path:

```
flyte run examples/genai/vllm/cache_model.py cache_model \
    --model_id Qwen/Qwen3-0.6B \
    --revision main
```

The output will be a flyte.io.Dir containing the model weights that can be used
as input to the VLLMAppEnvironment.
"""

import tempfile
from pathlib import Path
from typing import Optional

import flyte
import flyte.io

# Image with huggingface_hub for downloading models
image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "huggingface-hub[hf-transfer]>=0.25.0",
)

# TaskEnvironment for caching models from Hugging Face
cache_env = flyte.TaskEnvironment(
    name="cache-model-env",
    image=image,
    resources=flyte.Resources(cpu="4", memory="16Gi", disk="50Gi"),
    secrets="HF_TOKEN",  # Hugging Face token for private models
    env_vars={
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Enable fast downloads
    },
)


@cache_env.task(cache="auto")
async def cache_model(
    model_id: str = "Qwen/Qwen3-0.6B",
    revision: Optional[str] = None,
) -> flyte.io.Dir:
    """
    Download a model from Hugging Face Hub and save it as a Flyte directory.

    This task downloads all model files (weights, tokenizer, config) from
    Hugging Face and stores them in a remote blob store location that can
    be used by the VLLMAppEnvironment.

    Args:
        model_id: The Hugging Face model ID (e.g., "Qwen/Qwen3-0.6B").
        revision: Optional git revision (branch, tag, or commit hash).
            Defaults to the main branch.

    Returns:
        A flyte.io.Dir containing the downloaded model files.
    """
    from huggingface_hub import snapshot_download

    # Get HF token from secrets (optional, needed for private models)
    try:
        token = flyte.ctx().secrets.get("HF_TOKEN")
    except Exception:
        token = None
        print("No HF_TOKEN secret found, using anonymous access")

    # Create a temporary directory to download the model
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache_dir.mkdir(parents=True)

        print(f"Downloading model {model_id} (revision: {revision or 'main'})...")

        # Download the model snapshot from Hugging Face Hub
        local_model_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=str(cache_dir),
            local_files_only=False,
            token=token,
        )

        print(f"Model downloaded to: {local_model_path}")

        # List downloaded files
        model_files = list(Path(local_model_path).rglob("*"))
        print(f"Downloaded {len(model_files)} files")
        for f in model_files[:10]:  # Show first 10 files
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name}: {size_mb:.2f} MB")
        if len(model_files) > 10:
            print(f"  ... and {len(model_files) - 10} more files")

        # Upload the model directory to remote storage
        model_dir = await flyte.io.Dir.from_local(local_model_path)
        print(f"Model cached at: {model_dir.path}")

        return model_dir


@cache_env.task
async def main(
    model_id: str = "Qwen/Qwen3-0.6B",
    revision: Optional[str] = None,
) -> flyte.io.Dir:
    """
    Main entry point to cache a model.

    Args:
        model_id: The Hugging Face model ID.
        revision: Optional git revision.

    Returns:
        The cached model directory.
    """
    model_dir = await cache_model(model_id=model_id, revision=revision)
    print("Model successfully cached!")
    print(f"Model path: {model_dir.path}")
    print()
    print("To use this model with VLLMAppEnvironment, set MODEL_PATH to:")
    print(f"  export MODEL_PATH={model_dir.path}")
    return model_dir


if __name__ == "__main__":
    flyte.init_from_config()
    result = flyte.run(main, model_id="Qwen/Qwen3-0.6B")
    print(f"Cached model directory: {result}")
