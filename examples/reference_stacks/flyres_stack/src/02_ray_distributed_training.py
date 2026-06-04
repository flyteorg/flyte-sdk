"""
Flyte-Ray Plugin: Distributed Training with Hugging Face Model Mounts

This example shows how to use the Flyte-Ray plugin for distributed training,
combined with Hugging Face model mounts for efficient data access.
"""

# /// script
# requires-python = "==3.12"
# dependencies = [
#     "flyte",
#     "flyteplugins-ray",
#     "ray[default]>=2.40.0",
#     "transformers",
#     "datasets",
# ]
# ///

import os
from typing import Any, Dict

import ray
from flyteplugins.ray.task import HeadNodeConfig, RayJobConfig, WorkerNodeConfig

import flyte
from flyte import Image, Resources

# Configuration
HF_MODEL_PATH = os.environ.get("HF_MODEL_PATH", "/models")  # HF mount path
RAY_HEAD_PORT = 6379


def get_ray_image() -> Image:
    """Get the Ray plugin image with all necessary dependencies."""
    return (
        Image.from_debian_base(name="ray-training", python_version=(3, 12))
        .with_pip_packages(
            "flyteplugins-ray",
            "ray[default]==2.46.0",
            "torch==2.4.0",
            "transformers==4.42.0",
            "datasets",
            "wandb",  # For experiment tracking
        )
        .with_apt_packages("wget", "curl")
    )


# Ray cluster configuration
ray_config = RayJobConfig(
    head_node_config=HeadNodeConfig(ray_start_params={"log-color": "True"}),
    worker_node_config=[
        WorkerNodeConfig(group_name="training-group", replicas=3)  # 1 head + 3 workers
    ],
    runtime_env={
        "pip": ["torch", "transformers"],
        "env_vars": {
            "HF_HOME": HF_MODEL_PATH,
        },
    },
    enable_autoscaling=False,
    shutdown_after_job_finishes=True,
    ttl_seconds_after_finished=600,  # Keep cluster for 10 minutes
)


# Image and environments
ray_image = get_ray_image()
base_env = flyte.TaskEnvironment(
    name="ray_base",
    image=ray_image,
)
ray_env = flyte.TaskEnvironment(
    name="ray_training",
    plugin_config=ray_config,
    image=ray_image,
    resources=Resources(cpu=(4, 8), memory=("4Gi", "16Gi"), gpu="L4:1"),
)


@base_env.task
async def prepare_dataset() -> str:
    """
    Prepare and cache dataset to HF mount location.

    In production, this would download and preprocess a large dataset,
    storing it in the mounted Hugging Face model directory for fast access
    by all workers.
    """
    from datasets import Dataset

    # Create a simple synthetic dataset (in practice, use real data)
    data = {
        "text": [f"sample text {i}" for i in range(1000)],
        "label": [i % 2 for i in range(1000)],
    }
    Dataset.from_dict(data)

    # Save to HF mount (simulated - would save to actual path in production)
    output_path = os.path.join(HF_MODEL_PATH, "prepared_dataset")
    print(f"Dataset prepared and cached at: {output_path}")

    return output_path


@ray_env.task
async def distributed_train_and_evaluate(
    dataset_cache_path: str,
    num_epochs: int = 3,
) -> Dict[str, Any]:
    """
    Run distributed training using Ray with model from HF mount.

    This demonstrates:
    1. Accessing models from Hugging Face mount (shared storage)
    2. Distributing training across multiple nodes
    3. Aggregating results to Flyte
    """

    @ray.remote(num_gpus=1)
    class Trainer:
        def __init__(self, node_id: int):
            self.node_id = node_id

        def train_step(self, epoch: int) -> dict:
            """Simulate a training step."""

            # In production, this would:
            # 1. Load model from HF mount
            # 2. Train on local batch of data
            # 3. Return metrics

            return {
                "node_id": self.node_id,
                "epoch": epoch,
                "loss": 0.5 - (epoch * 0.1),  # Simulated decreasing loss
                "accuracy": 0.7 + (epoch * 0.05),
            }

        def get_node_info(self) -> dict:
            return {"node_id": self.node_id, "status": "ready"}

    # Initialize Ray if not already done
    if not ray.is_initialized():
        ray.init(address="auto")

    # Create trainers (one per worker)
    num_workers = 3
    trainers = [Trainer.remote(node_id=i) for i in range(num_workers)]

    # Run training epochs
    all_results = []
    for epoch in range(num_epochs):
        # Collect results from all workers
        futures = [trainer.train_step.remote(epoch) for trainer in trainers]
        epoch_results = ray.get(futures)
        all_results.extend(epoch_results)

        print(f"Epoch {epoch} complete. Results from {len(epoch_results)} workers.")

    # Compile final metrics
    avg_loss = sum(r["loss"] for r in all_results) / len(all_results)
    avg_accuracy = sum(r["accuracy"] for r in all_results) / len(all_results)

    return {
        "final_metrics": {
            "average_loss": round(avg_loss, 4),
            "average_accuracy": round(avg_accuracy, 4),
        },
        "total_epochs": num_epochs,
        "workers_used": num_workers,
        "dataset_cache_path": dataset_cache_path,
    }


@base_env.task
async def evaluate_model(results: Dict[str, Any]) -> str:
    """Evaluate the distributed training results."""
    metrics = results["final_metrics"]

    evaluation = {
        "status": "success",
        "metrics": metrics,
        "recommendation": "Model performance acceptable"
        if metrics["average_accuracy"] > 0.8
        else "Review training config",
    }

    return str(evaluation)


@ray_env.task
async def inference_serve(model_path: str, input_data: list) -> list:
    """
    Serve model predictions using Ray.

    Demonstrates how trained models from HF mounts can be served efficiently.
    """
    if not ray.is_initialized():
        ray.init(address="auto")

    @ray.remote(num_gpus=0.25)
    def predict(text: str) -> dict:
        """Simulate inference."""
        return {"text": text, "prediction": 1}

    # Run predictions in parallel
    futures = [predict.remote(text) for text in input_data]
    results = ray.get(futures)

    return results


async def main():
    """Run the complete distributed training workflow."""
    flyte.init_from_config()

    print("=" * 60)
    print("Flyte-Ray Distributed Training with HF Mounts")
    print("=" * 60)

    # Step 1: Prepare dataset
    print("\n[Step 1] Preparing dataset...")
    dataset_path = await flyte.run(prepare_dataset)
    print(f"Dataset cached at: {dataset_path.outputs[0]}")

    # Step 2: Run distributed training
    print("\n[Step 2] Running distributed training...")
    train_result = await flyte.run(distributed_train_and_evaluate, dataset_cache_path=dataset_path.outputs[0])
    print(f"Training complete. Results: {train_result.outputs[0]}")

    # Step 3: Evaluate results
    print("\n[Step 3] Evaluating results...")
    eval_result = await flyte.run(evaluate_model, results=train_result.outputs[0])
    print(f"Evaluation: {eval_result.outputs[0]}")

    # Step 4: Serve predictions
    print("\n[Step 4] Running inference...")
    sample_inputs = ["hello world", "test input", "inference example"]
    inference_result = await flyte.run(inference_serve, model_path="trained_model", input_data=sample_inputs)
    print(f"Inference results: {inference_result.outputs[0]}")


if __name__ == "__main__":
    # Uncomment to run locally (requires Ray to be installed and running)
    # asyncio.run(main())

    print("This example requires Flyte backend with Ray plugin enabled.")
    print("Run with: uv run --prerelease=allow src/02_ray_distributed_training.py")
