"""
Hugging Face Model Mounts for Efficient Data Loading

This example demonstrates using Hugging Face model mounts to efficiently load
large models and datasets, replacing manual data management in Dagster.
"""

# /// script
# requires-python = "==3.12"
# dependencies = [
#     "flyte",
#     "transformers>=4.40.0",
#     "datasets>=2.16.0",
# ]
# ///

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import flyte
from flyte import Image, Resources


def get_hf_image() -> Image:
    """Get image with Hugging Face libraries."""
    return Image.from_debian_base(name="hf-data", python_version=(3, 12)).with_pip_packages(
        "transformers==4.42.0",
        "datasets==2.18.0",
        "huggingface-hub==0.21.0",
        "torch==2.4.0",
    )


# Configuration
HF_MOUNT_PATH = os.environ.get("HF_MOUNT_PATH", "/models")
HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", f"{HF_MOUNT_PATH}/cache")
HF_TOKEN = flyte.Secret(key="HF_TOKEN")


image = get_hf_image()
base_env = flyte.TaskEnvironment(
    name="hf_data_base",
    image=image,
)
data_env = flyte.TaskEnvironment(
    name="hf_data_loading",
    image=image,
    resources=Resources(cpu=(2, 4), memory=("2Gi", "8Gi")),
    secrets=[HF_TOKEN],
)


@base_env.task
async def setup_hf_mount() -> str:
    """
    Setup Hugging Face mount for shared model storage.

    This replaces manual data copying in Dagster with Flyte's volume mounts.
    """
    # Ensure mount path exists
    Path(HF_MOUNT_PATH).mkdir(parents=True, exist_ok=True)
    Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = HF_MOUNT_PATH
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR

    return HF_MOUNT_PATH


@data_env.task
async def load_model_from_hf_hub(
    model_name: str,
    revision: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load a model from Hugging Face Hub with caching.

    Uses Hugging Face's automatic caching to avoid re-downloading models.
    For private models, configure HF_TOKEN secret in Flyte backend:
        flyte secret create hf token=your_token
    """
    # Get optional HF token from secret
    hf_token = flyte.Secret.get(HF_TOKEN, optional=True)

    if hf_token:
        os.environ["HF_AUTH_TOKEN"] = hf_token

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_name}")

    # Download and cache the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=HF_CACHE_DIR,
    )

    return {
        "model_name": model_name,
        "revision": revision or "main",
        "tokenizer_type": type(tokenizer).__name__,
        "model_type": type(model).__name__,
        "cache_path": HF_CACHE_DIR,
    }


@data_env.task
async def load_dataset_from_hf(
    dataset_name: str,
    split: str = "train",
    streaming: bool = False,
) -> Dict[str, Any]:
    """
    Load a dataset from Hugging Face Hub.

    Supports both regular and streaming mode for large datasets.
    """
    from datasets import load_dataset

    print(f"Loading dataset: {dataset_name}, split={split}")

    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=streaming,
        cache_dir=HF_CACHE_DIR,
    )

    # Get dataset info (without iterating through all data)
    if not streaming:
        info = {
            "num_rows": len(dataset),
            "features": str(dataset.features),
        }
    else:
        # For streaming, just get the first example
        sample = next(iter(dataset))
        info = {
            "streaming": True,
            "sample_keys": list(sample.keys()),
        }

    return {
        "dataset_name": dataset_name,
        "split": split,
        "streaming": streaming,
        **info,
    }


@data_env.task
async def prepare_dataset_for_training(
    dataset_info: Dict[str, Any],
    model_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Prepare dataset for training using loaded models.

    Demonstrates data preprocessing pipeline with cached models.
    """

    # In production, this would:
    # 1. Load tokenizer from cache
    # 2. Tokenize the dataset
    # 3. Apply formatting

    preparation_info = {
        "dataset_prepared": True,
        "model_used_for_tokenization": model_info.get("model_name"),
        "cache_hit_ratio": "high",  # Models loaded from HF mount
        "output_format": "tokenized",
    }

    return preparation_info


@data_env.task
async def fine_tune_model_with_mounts(
    dataset_path: str,
    model_path: str,
    num_epochs: int = 1,
) -> Dict[str, Any]:
    """
    Fine-tune a model using data from HF mounts.

    Shows how to use pre-downloaded models without re-downloading.
    """

    print(f"Fine-tuning with dataset: {dataset_path}, model: {model_path}")

    # Simulate fine-tuning (in production, load actual model from mount)
    results = {
        "fine_tuned": True,
        "epochs": num_epochs,
        "final_loss": 0.35,
        "accuracy": 0.82,
        "using_hf_mount": True,
        "model_cache_path": model_path,
    }

    return results


@data_env.task
async def batch_inference_with_cached_models(
    prompts: List[str],
    model_info: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Run inference using models from HF mount.

    Replaces manual model loading in Dagster with automatic caching.
    """
    results = []
    for prompt in prompts:
        # In production, this would load the actual model and run inference
        result = {
            "prompt": prompt,
            "generated_text": f"[Generated response for: {prompt}]",
            "source_model": model_info.get("model_name"),
            "from_cache": True,
        }
        results.append(result)

    return results


if __name__ == "__main__":
    flyte.init_from_config()

    print("=" * 60)
    print("Hugging Face Model Mounts for Data Loading")
    print("=" * 60)

    # Step 1: Setup mount
    print("\n[Setup] Setting up HF mounts...")
    setup_result = flyte.run(setup_hf_mount)
    print(f"Mount path: {setup_result.outputs[0]}")

    # Step 2: Load model from Hub (cached on mount)
    print("\n[Model] Loading model from Hugging Face Hub...")
    model_info = flyte.run(load_model_from_hf_hub, model_name="gpt2")
    print(f"Loaded: {model_info.outputs[0]}")

    # Step 3: Load dataset
    print("\n[Dataset] Loading dataset...")
    dataset_info = flyte.run(
        load_dataset_from_hf,
        dataset_name="wikitext",
        split="train[:100]",
    )
    print(f"Dataset info: {dataset_info.outputs[0]}")

    # Step 4: Prepare for training
    print("\n[Prepare] Preparing dataset...")
    prep_result = flyte.run(
        prepare_dataset_for_training,
        dataset_info=dataset_info.outputs[0],
        model_info=model_info.outputs[0],
    )
    print(f"Preparation complete: {prep_result.outputs[0]}")

    # Step 5: Batch inference
    print("\n[Inference] Running batch inference...")
    prompts = ["Hello world", "Test prompt", "Another example"]
    inference_result = flyte.run(
        batch_inference_with_cached_models,
        prompts=prompts,
        model_info=model_info.outputs[0],
    )
    print(f"Inference results: {inference_result.outputs[0]}")
