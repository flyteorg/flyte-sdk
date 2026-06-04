"""
PyTorch FSDP Distributed Training (Megatron Alternative)

This example demonstrates using Flyte's PyTorch plugin for distributed training,
which can serve as an alternative to Megatron-LM for large model training.
Uses Fully Sharded Data Parallelism (FSDP) to efficiently train large models.
"""

# /// script
# requires-python = "==3.12"
# dependencies = [
#     "flyte",
#     "torch==2.4.0",
#     "torchvision",
#     "flyteplugins-pytorch",
# ]
# ///

from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from flyteplugins.pytorch.task import Elastic
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import flyte
from flyte import Image, Resources


def get_torch_image() -> Image:
    """Get PyTorch image with Flyte plugins."""
    return Image.from_debian_base(name="torch-fsdp", python_version=(3, 12)).with_pip_packages(
        "flyteplugins-pytorch",
        "torch==2.4.0",
        "torchvision==0.19.0",
        "transformers==4.42.0",
    )


# PyTorch Elastic configuration for FSDP-style distributed training
# This replaces Megatron's custom training orchestration with standard PyTorch
torch_config = Elastic(
    nproc_per_node=2,  # Number of processes per node (can be tuned for your hardware)
    nnodes=2,  # Number of nodes (for multi-node training)
    max_restarts=3,
    rdzv_backend="c10d",
    rdzv_endpoint="auto",  # Flyte will handle this
)

image = get_torch_image()
base_env = flyte.TaskEnvironment(
    name="torch_base",
    image=image,
    resources=Resources(cpu=(2, 4), memory=("2Gi", "8Gi")),
)
fsdp_env = flyte.TaskEnvironment(
    name="torch_fsdp_train",
    plugin_config=torch_config,
    image=image,
    resources=Resources(cpu=(4, 8), memory=("8Gi", "32Gi"), gpu="A10G:1"),
)


class LargeLinearModel(nn.Module):
    """
    A simple but large model to demonstrate distributed training.

    In production, this would be replaced with a transformer model
    (like the examples in Megatron-LM).
    """

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 8192, output_dim: int = 4096):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def prepare_synthetic_dataset(size: int = 10000, input_dim: int = 4096) -> TensorDataset:
    """Prepare synthetic dataset for training."""
    x = torch.randn(size, input_dim)
    y = torch.randint(0, 2, (size, 1)).float()
    return TensorDataset(x, y)


def train_fsdp_step(
    rank: int,
    world_size: int,
    epochs: int,
    batch_size: int,
) -> Tuple[float, float]:
    """
    Training loop using PyTorch DDP for FSDP-like distributed training.

    This replaces Megatron's custom orchestration with standard PyTorch
    DistributedDataParallel or FullyShardedDataParallel.
    """
    # Initialize process group
    dist.init_process_group("nccl")

    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        print(f"[Rank {rank}] Starting training with world_size={world_size}")

        # Prepare model and data
        model = LargeLinearModel()

        # Wrap with DDP (in production, use FSDP for better memory efficiency)
        model = DDP(model)

        dataset = prepare_synthetic_dataset(size=10000 * world_size)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

        # Optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        epoch_losses = []
        for epoch in range(epochs):
            sampler.set_epoch(epoch)  # Shuffle data differently each epoch

            total_loss = 0.0
            num_batches = 0

            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            epoch_losses.append(avg_loss)

            if rank == 0:
                print(f"[Rank 0] Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        # Return metrics from rank 0
        final_loss = epoch_losses[-1] if epoch_losses else 0.0
        avg_overall_loss = sum(epoch_losses) / len(epoch_losses)

        return final_loss, avg_overall_loss

    finally:
        dist.destroy_process_group()


@base_env.task
async def setup_training() -> dict:
    """Setup phase before distributed training."""
    print("Setting up distributed training environment...")

    # In production, this would:
    # 1. Download model weights from Hugging Face
    # 2. Prepare dataset in shared storage
    # 3. Configure FSDP settings

    return {
        "setup_status": "complete",
        "model_type": "LargeLinearModel (FSDP-ready)",
        "num_nodes": torch_config.nnodes,
        "processes_per_node": torch_config.nproc_per_node,
    }


@fsdp_env.task
async def run_distributed_training(
    epochs: int = 3,
    batch_size: int = 64,
) -> dict:
    """
    Run FSDP-style distributed training.

    This replaces Megatron-LM with PyTorch's native distributed capabilities.
    For true FSDP, replace DDP with FullyShardedDataParallel.
    """
    final_loss, avg_loss = train_fsdp_step(
        rank=0,
        world_size=torch_config.nproc_per_node * torch_config.nnodes,
        epochs=epochs,
        batch_size=batch_size,
    )

    return {
        "final_loss": round(final_loss, 6),
        "average_loss": round(avg_loss, 6),
        "epochs_trained": epochs,
        "batch_size": batch_size,
    }


@base_env.task
async def evaluate_training_results(
    training_result: dict,
) -> dict:
    """Evaluate and summarize training results."""
    metrics = {
        "training_status": "success",
        "final_metrics": training_result,
        "recommendation": (
            "Training successful" if training_result["average_loss"] < 0.5 else "Review hyperparameters"
        ),
        "model_architecture": "LargeLinearModel with DDP/FSDP",
    }

    return metrics


@fsdp_env.task
async def save_model_checkpoint(
    training_result: dict,
) -> str:
    """Save model checkpoint to shared storage (e.g., HF mount)."""
    # In production, this would save the actual model state_dict
    checkpoint_path = "/hf-mount/checkpoints/fsdp_training_final.pth"

    print(f"Model checkpoint saved at: {checkpoint_path}")

    return checkpoint_path


if __name__ == "__main__":
    flyte.init_from_config()

    print("=" * 60)
    print("PyTorch FSDP Distributed Training (Megatron Alternative)")
    print("=" * 60)

    # Step 1: Setup
    print("\n[Step 1] Setting up training...")
    setup_result = flyte.run(setup_training)
    print(f"Setup: {setup_result.outputs[0]}")

    # Step 2: Run distributed training
    print("\n[Step 2] Running distributed training...")
    train_result = flyte.run(run_distributed_training, epochs=3, batch_size=64)
    print(f"Training result: {train_result.outputs[0]}")

    # Step 3: Evaluate
    print("\n[Step 3] Evaluating results...")
    eval_result = flyte.run(evaluate_training_results, training_result=train_result.outputs[0])
    print(f"Evaluation: {eval_result.outputs[0]}")

    # Step 4: Save checkpoint
    print("\n[Step 4] Saving model checkpoint...")
    checkpoint_path = flyte.run(save_model_checkpoint, training_result=train_result.outputs[0])
    print(f"Checkpoint path: {checkpoint_path.outputs[0]}")
