"""
Experiment Tracking with Weights & Biases

This example shows how to integrate W&B for experiment tracking within Flyte workflows,
replacing Dagster's custom tracking mechanisms.

Secrets must be configured in the Flyte backend:
- wandb/api-key: Your W&B API key
- wandb/entity: (optional) Your W&B entity/team name
"""

# /// script
# requires-python = "==3.12"
# dependencies = [
#     "flyte",
#     "wandb>=0.16.0",
# ]
# ///

import os
from typing import Any, Dict, List

import wandb
from flyteplugins.ray.task import HeadNodeConfig, RayJobConfig, WorkerNodeConfig

import flyte
from flyte import Image


def get_tracking_image() -> Image:
    """Get image with W&B support."""
    return Image.from_debian_base(name="experiment-tracking", python_version=(3, 12)).with_pip_packages(
        "wandb==0.17.0",
        "torch==2.4.0",
        "transformers==4.42.0",
    )


# W&B configuration - secrets must be configured in Flyte backend
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "flyte-experiments")

# Define secrets (these would be created via flyteadmin CLI or dashboard)
_wandb_api_key_secret = flyte.Secret(
    name="wandb",
    key="api-key",
    group="experiment-tracking",
    description="W&B API key for experiment tracking",
)
_wandb_entity_secret = flyte.Secret(
    name="wandb",
    key="entity",
    group="experiment-tracking",
    optional=True,
    description="W&B entity/team name (optional)",
)

image = get_tracking_image()
base_env = flyte.TaskEnvironment(
    name="tracking_base",
    image=image,
)
ray_env = flyte.TaskEnvironment(
    name="distributed_training_with_tracking",
    plugin_config=RayJobConfig(
        head_node_config=HeadNodeConfig(),
        worker_node_config=[WorkerNodeConfig(group_name="training", replicas=2)],
        runtime_env={"env_vars": {"WANDB_MODE": "offline"}},
    ),
    image=image,
)


@base_env.task
async def initialize_wandb_run(
    experiment_name: str,
    hyperparameters: Dict[str, Any],
) -> str:
    """
    Initialize a W&B run for tracking.

    Replaces Dagster's custom run metadata system with W&B's experiment tracking.
    """
    # Get API key from secret
    wandb_api_key = flyte.Secret.get(_wandb_api_key_secret)
    wandb_entity = flyte.Secret.get(_wandb_entity_secret, optional=True)

    # Set up environment variables for W&B
    os.environ["WANDB_API_KEY"] = wandb_api_key
    if wandb_entity:
        os.environ["WANDB_ENTITY"] = wandb_entity

    # Initialize W&B run
    run = wandb.init(
        project=WANDB_PROJECT,
        name=experiment_name,
        config=hyperparameters,
        tags=["flyte", "training"],
    )

    print(f"Initialized W&B run: {run.name} ({run.id})")

    return run.id


@base_env.task
async def log_metrics(run_id: str, metrics: Dict[str, float], step: int) -> None:
    """
    Log training metrics to W&B.

    Replaces Dagster's custom metric logging with W&B's native integration.
    """
    wandb.init(id=run_id, project=WANDB_PROJECT, resume="allow")
    wandb.log(metrics, step=step)
    print(f"Logged metrics at step {step}: {metrics}")


@base_env.task
async def log_model(run_id: str, model_path: str, model_name: str) -> dict:
    """
    Log a model artifact to W&B.

    Provides versioning and lineage tracking for models.
    """
    wandb.init(id=run_id, project=WANDB_PROJECT, resume="allow")

    # Log model as artifact
    artifact = wandb.Artifact(model_name, type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    return {
        "artifact_name": model_name,
        "run_id": run_id,
        "logged_at_step": wandb.run.step if wandb.run else 0,
    }


@base_env.task
async def finish_wandb_run(run_id: str, final_metrics: Dict[str, float]) -> dict:
    """
    Finish a W&B run and log final metrics.

    Replaces Dagster's pipeline completion hooks with W&B's run termination.
    """
    wandb.init(id=run_id, project=WANDB_PROJECT, resume="allow")

    # Log final metrics
    wandb.log(final_metrics)
    wandb.finish()

    return {
        "run_id": run_id,
        "final_metrics": final_metrics,
        "status": "completed",
    }


@ray_env.task
async def train_with_tracking(
    experiment_name: str,
    hyperparameters: Dict[str, Any],
    epochs: int = 3,
) -> dict:
    """
    Train a model with W&B integration.

    This replaces Dagster's op-level tracking with W&B's native tracking.
    """
    import torch

    # Initialize tracking
    run_id = await initialize_wandb_run(experiment_name, hyperparameters)

    # Simulate training loop
    metrics_history = []

    for epoch in range(epochs):
        # Simulate training metrics (in production, compute real metrics)
        train_loss = max(0.1, 1.0 - epoch * 0.2)  # Decreasing loss
        val_loss = max(0.15, 1.2 - epoch * 0.25)
        accuracy = min(0.95, 0.5 + epoch * 0.15)

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy,
            "learning_rate": hyperparameters.get("lr", 0.001),
        }

        # Log to W&B
        await log_metrics(run_id, metrics, step=epoch)
        metrics_history.append(metrics)

    # Final evaluation
    final_metrics = {
        "final_train_loss": metrics_history[-1]["train_loss"],
        "final_val_loss": metrics_history[-1]["val_loss"],
        "final_accuracy": metrics_history[-1]["accuracy"],
        "epochs_trained": epochs,
    }

    # Log model artifact
    checkpoint_path = "/tmp/model_checkpoint.pth"
    torch.save({"epoch": epochs, "state_dict": {}}, checkpoint_path)
    await log_model(run_id, checkpoint_path, f"{experiment_name}_model")

    # Finish run
    result = await finish_wandb_run(run_id, final_metrics)

    return {
        **result,
        "metrics_history": metrics_history,
    }


@base_env.task
async def compare_experiments(
    experiment_results: List[dict],
) -> dict:
    """
    Compare multiple experiments and select the best one.

    Replaces Dagster's custom comparison logic with W&B's experiment comparison.
    """
    if not experiment_results:
        return {"error": "No experiments to compare"}

    # Sort by accuracy (or any other metric)
    best_experiment = max(experiment_results, key=lambda x: x.get("final_metrics", {}).get("final_accuracy", 0))

    comparison = {
        "total_experiments": len(experiment_results),
        "best_experiment_run_id": best_experiment["run_id"],
        "best_accuracy": best_experiment["final_metrics"].get("final_accuracy"),
        "all_runs": [r["run_id"] for r in experiment_results],
    }

    return comparison


async def run_example_workflow():
    """Run the complete experiment tracking workflow."""
    flyte.init_from_config()

    print("=" * 60)
    print("Experiment Tracking with W&B")
    print("=" * 60)

    # Configure experiments
    hyperparams1 = {"lr": 0.001, "batch_size": 32, "model_type": "small"}
    hyperparams2 = {"lr": 0.0005, "batch_size": 64, "model_type": "medium"}

    # Run experiment 1
    print("\n[Experiment 1] Training with lr=0.001...")
    exp1_result = await flyte.run(
        train_with_tracking,
        experiment_name="exp_lr_001",
        hyperparameters=hyperparams1,
        epochs=3,
    )
    print(f"Run ID: {exp1_result.outputs[0]['run_id']}")

    # Run experiment 2
    print("\n[Experiment 2] Training with lr=0.0005...")
    exp2_result = await flyte.run(
        train_with_tracking,
        experiment_name="exp_lr_0005",
        hyperparameters=hyperparams2,
        epochs=3,
    )
    print(f"Run ID: {exp2_result.outputs[0]['run_id']}")

    # Compare experiments
    print("\n[Compare] Comparing experiments...")
    comparison = await flyte.run(
        compare_experiments,
        experiment_results=[exp1_result.outputs[0], exp2_result.outputs[0]],
    )
    print(f"Comparison: {comparison.outputs[0]}")


if __name__ == "__main__":
    # Uncomment to run locally (requires W&B API key and backend)
    # asyncio.run(run_example_workflow())

    print("Experiment tracking example")
    print("===========================")
    print()
    print("This example requires:")
    print("1. W&B API key configured as Flyte secret: flyte secret create wandb api-key=your_key")
    print(f"2. Project name (default: {WANDB_PROJECT})")
    print("3. Optional: entity secret - flyte secret create wandb entity=my_entity")
    print()
    print("Run with: uv run --prerelease=allow src/04_experiment_tracking.py")
