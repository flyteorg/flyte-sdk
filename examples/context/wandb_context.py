"""
Flyte + Weights & Biases Integration Example

This example demonstrates how to use W&B with Flyte for experiment tracking.
The Wandb link integration creates clickable links in the Flyte UI to view your experiment results.
"""

import os
import random
from dataclasses import dataclass

import wandb

import flyte
from flyte.link import Wandb

# Configure the task environment with wandb
env = flyte.TaskEnvironment(
    "wandb-example",
    image=flyte.Image.from_debian_base().with_pip_packages("wandb"),
    secrets="WANDB_API_KEY",
)

# W&B project configuration - update these with your own values
WANDB_PROJECT = "flyte-ml-experiments"
WANDB_ENTITY = "pingsutw"  # Your W&B username or team name


@dataclass
class ModelConfig:
    """Configuration for model training."""

    learning_rate: float
    batch_size: int
    epochs: int
    model_name: str = "simple-classifier"


@dataclass
class TrainingResult:
    """Results from a training run."""

    final_val_acc: float
    final_val_loss: float
    best_epoch: int


@env.task()
def train_model(config: ModelConfig) -> TrainingResult:
    """
    Training task that runs a single experiment and logs metrics to W&B.

    The @env.task decorator with Wandb link creates a clickable link in Flyte UI
    that takes you directly to the W&B run for this specific training job.

    Args:
        config: Model configuration with hyperparameters

    Returns:
        Training results including final accuracy and best epoch
    """
    with wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        id=flyte.ctx().custom_context.get("wandb_id"),
        config={
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "model_name": config.model_name,
        },
    ) as run:
        print(
            f"Training {config.model_name} with lr={config.learning_rate}, "
            f"batch_size={config.batch_size}, epochs={config.epochs}"
        )
        print(f"W&B Run URL: {run.url}")

        best_val_acc = 0.0
        best_epoch = 0
        val_acc = 0.0
        val_loss = 0.0

        # Simulate training loop
        for epoch in range(1, config.epochs + 1):
            # Simulate training metrics (replace with actual model training)
            train_acc = 0.3 + (epoch / config.epochs * 0.6) + random.random() * 0.1
            train_loss = 0.7 - (epoch / config.epochs * 0.5) - random.random() * 0.1

            # Simulate validation metrics
            val_acc = 0.25 + (epoch / config.epochs * 0.65) + random.random() * 0.08
            val_loss = 0.75 - (epoch / config.epochs * 0.55) - random.random() * 0.08

            # Track best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch

            # Log metrics to W&B
            wandb.log(
                {
                    "epoch": epoch,
                    "train/accuracy": train_acc,
                    "train/loss": train_loss,
                    "val/accuracy": val_acc,
                    "val/loss": val_loss,
                    "best_val_accuracy": best_val_acc,
                }
            )

            print(f"Epoch {epoch}/{config.epochs} - train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}")

        # Log final summary metrics
        run.summary.update(
            {
                "final_val_accuracy": val_acc,
                "final_val_loss": val_loss,
                "best_val_accuracy": best_val_acc,
                "best_epoch": best_epoch,
            }
        )

        print(f"Training complete! Best val_acc: {best_val_acc:.4f} at epoch {best_epoch}")

    return TrainingResult(
        final_val_acc=val_acc,
        final_val_loss=val_loss,
        best_epoch=best_epoch,
    )


@env.task()
async def main() -> TrainingResult:
    """
    Main task that runs a single training experiment with W&B tracking.

    This demonstrates how to use Flyte with W&B for experiment tracking.
    The train_model task will have a clickable link in the Flyte UI to view the W&B run.
    """
    # Define the configuration for this training run
    config = ModelConfig(learning_rate=0.001, batch_size=32, epochs=10, model_name="simple-classifier")

    print("Starting training experiment with configuration:")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Model: {config.model_name}")

    # Use custom context to track W&B run ID
    wandb_id = os.getenv("_F_PN", f"flyte-run-{random.randint(1000, 9999)}")

    with flyte.custom_context(wandb_id=wandb_id):
        result = train_model.override(link=Wandb(project=WANDB_PROJECT, entity=WANDB_ENTITY, id=wandb_id))(
            config=config
        )

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Final validation accuracy: {result.final_val_acc:.4f}")
    print(f"Final validation loss: {result.final_val_loss:.4f}")
    print(f"Best epoch: {result.best_epoch}")
    print(f"View run at: https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}")
    print(f"{'=' * 60}")

    return result


if __name__ == "__main__":
    # Initialize Flyte and run the training
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote").run(main)
    run.wait()

    print("\n" + "=" * 60)
    print("✅ Experiment complete!")
    print(f"Flyte execution URL: {run.url}")
    print(f"W&B project: https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}")
    print("=" * 60)
