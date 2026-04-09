# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte",
# ]
# ///
"""
LLaMA-Factory Fine-Tuning on Union

Fine-tune Qwen3-0.6B on a HuggingFace Q&A dataset using LLaMA-Factory's LoRA
training, orchestrated as a Union workflow.

LLaMA-Factory (https://github.com/hiyouga/LlamaFactory) is a framework for
efficient fine-tuning of 100+ LLMs — it supports LoRA, QLoRA, DPO, PPO, and
more via simple YAML configs.

This example:
1. Downloads the PHD-Science subset of ianncity/KIMI-K2.5-1000000x from
   HuggingFace and converts it to LLaMA-Factory's expected format
2. Trains a LoRA adapter on Qwen3-0.6B using llamafactory-cli
3. Optionally merges the adapter into the base model for deployment
4. Evaluates the fine-tuned model on a few science prompts to verify it works

Prerequisites
-------------
- A Union account with GPU access (L4 or better)
- A HuggingFace token stored as a Union secret named "hf-token"

Run
---
uv run --prerelease=allow examples/genai/llamafactory_finetune.py
"""

import json
import subprocess
import tempfile
from pathlib import Path

import flyte
import flyte.io

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------

gpu_image = (
    flyte.Image.from_debian_base(name="llamafactory", python_version=(3, 12))
    .with_apt_packages("git", "build-essential")
    .with_pip_packages(
        "flyte>=2.1.4",
        "llamafactory[torch,metrics]",
        "transformers>=4.46.0",
        "accelerate>=0.34.0",
        "peft>=0.13.0",
        "trl>=0.12.0",
        "datasets>=3.1.0",
        "huggingface_hub",
    )
)

# ---------------------------------------------------------------------------
# Task environments
# ---------------------------------------------------------------------------

data_env = flyte.TaskEnvironment(
    name="data_prep",
    image=gpu_image,
    resources=flyte.Resources(cpu=4, memory="16Gi", disk="50Gi"),
    secrets=[flyte.Secret(key="hf-token", as_env_var="HF_TOKEN")],
    cache="auto",
)

training_env = flyte.TaskEnvironment(
    name="llamafactory_training",
    image=gpu_image,
    resources=flyte.Resources(
        cpu=6,
        memory="24Gi",
        gpu="L4:1",
        disk="100Gi",
        shm="4Gi",
    ),
    secrets=[flyte.Secret(key="hf-token", as_env_var="HF_TOKEN")],
)

driver_env = flyte.TaskEnvironment(
    name="llamafactory_driver",
    image=gpu_image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[data_env, training_env],
)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@data_env.task
async def prepare_dataset(
    hf_dataset: str,
    hf_subset: str,
    max_samples: int,
) -> flyte.io.Dir:
    """
    Download a HuggingFace dataset and convert it to LLaMA-Factory format.

    The dataset must have a "messages" column with role/content dicts
    (sharegpt format). We write:
    - data/train.json with the conversation data
    - data/dataset_info.json pointing LLaMA-Factory at it
    """
    from datasets import load_dataset

    print(f"Loading {hf_dataset} (subset={hf_subset}, max_samples={max_samples})...")
    ds = load_dataset(hf_dataset, hf_subset, split="train")

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    print(f"Loaded {len(ds)} samples")

    output_dir = Path("/tmp/llamafactory_data")
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Convert to LLaMA-Factory sharegpt format:
    # [{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}]
    records = []
    for row in ds:
        messages = row["messages"]
        conversations = []
        for msg in messages:
            role_map = {"user": "human", "assistant": "gpt", "system": "system"}
            conversations.append({
                "from": role_map.get(msg["role"], msg["role"]),
                "value": msg["content"],
            })
        records.append({"conversations": conversations})

    train_path = data_dir / "train.json"
    with open(train_path, "w") as f:
        json.dump(records, f, ensure_ascii=False)

    # Write dataset_info.json so llamafactory-cli can find our data
    dataset_info = {
        "custom_train": {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
            },
        }
    }
    with open(data_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Wrote {len(records)} records to {train_path}")
    return await flyte.io.Dir.from_local(str(output_dir))


@training_env.task
async def train_lora(
    data_dir: flyte.io.Dir,
    model_name: str,
    template: str,
    lora_rank: int,
    num_train_epochs: float,
    learning_rate: float,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
) -> flyte.io.Dir:
    """Run LoRA fine-tuning via llamafactory-cli."""
    import yaml

    data_path = data_dir.download_sync()
    output_dir = "/tmp/lora_output"

    train_config = {
        # Model
        "model_name_or_path": model_name,
        "trust_remote_code": True,
        # Method
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": lora_rank,
        "lora_target": "all",
        # Dataset
        "dataset": "custom_train",
        "dataset_dir": str(Path(data_path) / "data"),
        "template": template,
        "cutoff_len": 1024,
        "preprocessing_num_workers": 4,
        "dataloader_num_workers": 2,
        # Output
        "output_dir": output_dir,
        "logging_steps": 10,
        "save_steps": 500,
        "overwrite_output_dir": True,
        "save_only_model": True,
        "report_to": "none",
        # Training
        "per_device_train_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
    }

    config_path = Path(tempfile.mkdtemp()) / "train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(train_config, f, default_flow_style=False)

    print(f"Training config:\n{yaml.dump(train_config, default_flow_style=False)}")
    print("Starting LLaMA-Factory training...")

    subprocess.run(
        ["llamafactory-cli", "train", str(config_path)],
        check=True,
    )

    print("Training completed!")
    return await flyte.io.Dir.from_local(output_dir)


@training_env.task
async def export_merged_model(
    adapter: flyte.io.Dir,
    model_name: str,
    template: str,
) -> flyte.io.Dir:
    """Merge LoRA adapter weights into the base model."""
    import yaml

    adapter_path = adapter.download_sync()
    export_dir = "/tmp/merged_model"

    export_config = {
        "model_name_or_path": model_name,
        "adapter_name_or_path": str(adapter_path),
        "template": template,
        "finetuning_type": "lora",
        "trust_remote_code": True,
        "export_dir": export_dir,
        "export_size": 2,
        "export_legacy_format": False,
    }

    config_path = Path(tempfile.mkdtemp()) / "export_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(export_config, f, default_flow_style=False)

    print("Merging LoRA adapter into base model...")

    subprocess.run(
        ["llamafactory-cli", "export", str(config_path)],
        check=True,
    )

    print(f"Merged model exported to {export_dir}")
    return await flyte.io.Dir.from_local(export_dir)


TEST_PROMPTS = [
    "Explain the process of photosynthesis.",
    "What is the Heisenberg uncertainty principle?",
    "Describe how CRISPR-Cas9 works.",
]


@training_env.task
async def evaluate_model(
    model_dir: flyte.io.Dir,
    prompts: list[str] = TEST_PROMPTS,
    max_new_tokens: int = 200,
) -> str:
    """Load the fine-tuned model and run inference on test prompts."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = model_dir.download_sync()

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    results = []
    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*60}")

        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        print(f"RESPONSE: {response}")
        results.append(f"Q: {prompt}\nA: {response}")

    full_results = "\n\n---\n\n".join(results)
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    return full_results


@driver_env.task
async def finetune_pipeline(
    hf_dataset: str = "ianncity/KIMI-K2.5-1000000x",
    hf_subset: str = "PHD-Science",
    max_samples: int = 500,
    model_name: str = "Qwen/Qwen3-0.6B",
    template: str = "qwen3",
    num_epochs: float = 1.0,
    lora_rank: int = 8,
    learning_rate: float = 1e-4,
    export_merged: bool = True,
) -> tuple[flyte.io.Dir, str]:
    """
    End-to-end LLM fine-tuning pipeline using LLaMA-Factory.

    Downloads a dataset from HuggingFace, fine-tunes a LoRA adapter on Qwen3,
    and optionally merges the adapter back into the base model.

    Args:
        hf_dataset: HuggingFace dataset ID
        hf_subset: Dataset subset/config name
        max_samples: Cap on training samples (for quick experiments)
        model_name: HuggingFace model ID
        template: LLaMA-Factory chat template name
        num_epochs: Number of training epochs
        lora_rank: LoRA rank (higher = more capacity, more memory)
        learning_rate: Peak learning rate
        export_merged: Whether to merge LoRA into the base model
    """
    data = await prepare_dataset(
        hf_dataset=hf_dataset,
        hf_subset=hf_subset,
        max_samples=max_samples,
    )

    adapter = await train_lora(
        data_dir=data,
        model_name=model_name,
        template=template,
        lora_rank=lora_rank,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_batch_size=2,
        gradient_accumulation_steps=4,
    )

    if export_merged:
        model = await export_merged_model(
            adapter=adapter,
            model_name=model_name,
            template=template,
        )
    else:
        model = adapter

    eval_results = await evaluate_model(model_dir=model)
    return model, eval_results


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(
        finetune_pipeline,
        max_samples=100,
        num_epochs=1.0,
    )
    print(f"Run URL: {run.url}")
