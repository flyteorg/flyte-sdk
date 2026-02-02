"""
Example: Distributed Training with W&B Integration

This example demonstrates all distributed training scenarios with W&B logging:

1. Single-node multi-GPU:
   - run_mode="auto" (default): Only rank 0 logs → 1 W&B run
   - run_mode="shared": All GPUs log to 1 shared run
   - run_mode="new": Each GPU gets its own run (grouped)

2. Multi-node multi-GPU:
   - run_mode="auto" (default): Local rank 0 of each worker logs → N W&B runs (1 per worker)
   - run_mode="shared": All GPUs per worker log to shared run → N W&B runs (1 per worker)
   - run_mode="new": Each GPU gets its own run (grouped per worker) → NxGPUs W&B runs

Run ID patterns:
- Single-node auto: {run_name}-{action_name}
- Single-node shared: {run_name}-{action_name}
- Single-node new: {run_name}-{action_name}-rank-{rank} (grouped)
- Multi-node auto: {run_name}-{action_name}-worker-{worker_index}
- Multi-node shared: {run_name}-{action_name}-worker-{worker_index}
- Multi-node new: {run_name}-{action_name}-worker-{worker_index}-rank-{local_rank} (grouped per worker)

Example command:
uv run example_distributed.py single_node_auto
"""

import typing

import flyte
import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim
from flyteplugins.pytorch.task import Elastic
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from flyteplugins.wandb import (
    get_distributed_info,
    get_wandb_run,
    wandb_config,
    wandb_init,
)

image = flyte.Image.from_debian_base(name="torch-wandb").with_pip_packages(
    "flyteplugins-wandb", "flyteplugins-pytorch", pre=True
)

# Single-node environment (1 node, 4 GPUs)
single_node_env = flyte.TaskEnvironment(
    name="single_node_env",
    resources=flyte.Resources(cpu=(1, 2), memory=("1Gi", "10Gi"), gpu="V100:4", shm="auto"),
    plugin_config=Elastic(
        nproc_per_node=4,
        nnodes=1,
    ),
    image=image,
    secrets=flyte.Secret(key="wandb_api_key", as_env_var="WANDB_API_KEY"),
)

# Multi-node environment (2 nodes, 4 GPUs each)
multi_node_env = flyte.TaskEnvironment(
    name="multi_node_env",
    resources=flyte.Resources(cpu=(1, 2), memory=("1Gi", "10Gi"), gpu="V100:4", shm="auto"),
    plugin_config=Elastic(
        nproc_per_node=4,
        nnodes=2,
    ),
    image=image,
    secrets=flyte.Secret(key="wandb_api_key", as_env_var="WANDB_API_KEY"),
)


class MLP(nn.Module):
    """Multi-layer perceptron for image classification."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 512, num_classes: int = 10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


class SyntheticDataset(torch.utils.data.Dataset):
    """Synthetic dataset simulating image classification data."""

    def __init__(self, num_samples: int = 50000, input_dim: int = 784, num_classes: int = 10):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        # Pre-generate data for consistency across epochs
        self.data = torch.randn(num_samples, input_dim)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def _train_loop_impl(duration_seconds: int = 300) -> float | None:
    """Core training loop. Runs for specified duration."""
    import time

    torch.distributed.init_process_group("nccl")

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(torch.distributed.get_rank() % 4)

    # Set device
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Model
    model = MLP(input_dim=784, hidden_dim=512, num_classes=10).to(device)
    model = DDP(model, device_ids=[local_rank])

    # Dataset and dataloader
    dataset = SyntheticDataset(num_samples=50000, input_dim=784, num_classes=10)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    # W&B
    wandb_run = get_wandb_run()
    dist_info = get_distributed_info()

    print(f"[Rank {rank}] Distributed info: {dist_info}")
    print(f"[Rank {rank}] wandb_run: {wandb_run}")

    start_time = time.time()
    global_step = 0
    epoch = 0
    running_loss = 0.0
    num_batches = 0

    print(
        f"[Rank {rank}] Starting training for {duration_seconds}s | "
        f"Dataset: {len(dataset)} samples | Batch size: 128 | Workers: {world_size}"
    )

    while time.time() - start_time < duration_seconds:
        sampler.set_epoch(epoch)
        model.train()

        for batch_idx, (d, t) in enumerate(dataloader):
            data, target = d.to(device), t.to(device)

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Log every 100 steps
            if global_step % 100 == 0:
                avg_loss = running_loss / num_batches
                elapsed = time.time() - start_time
                throughput = global_step * 128 * world_size / elapsed

                if wandb_run:
                    wandb_run.log(
                        {
                            "train/loss": avg_loss,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                            "train/global_step": global_step,
                            "train/throughput_samples_per_sec": throughput,
                            "rank": rank,
                            "local_rank": local_rank,
                            "worker_index": (dist_info["worker_index"] if dist_info else 0),
                        }
                    )

                if local_rank == 0:
                    print(
                        f"[Step {global_step}] Loss: {avg_loss:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"Throughput: {throughput:.0f} samples/s | "
                        f"Elapsed: {elapsed:.0f}s"
                    )

                running_loss = 0.0
                num_batches = 0

            if time.time() - start_time >= duration_seconds:
                break

        epoch += 1

    total_time = time.time() - start_time
    if local_rank == 0:
        print(
            f"[Worker {dist_info['worker_index'] if dist_info else 0}] Training complete | "
            f"Epochs: {epoch} | Steps: {global_step} | Time: {total_time:.1f}s"
        )

    # Barrier to ensure all ranks finish training before cleanup
    # This prevents one worker from destroying the process group while another is still training
    print(f"[Rank {rank}] Waiting at barrier...")
    torch.distributed.barrier()
    print(f"[Rank {rank}] Barrier passed, calling destroy_process_group...")
    torch.distributed.destroy_process_group()
    print(f"[Rank {rank}] destroy_process_group done")

    print(running_loss / max(num_batches, 1))

    # Each worker's primary (local_rank 0) returns a value
    # In multi-node, this means each worker returns independently
    if local_rank == 0:
        return running_loss / max(num_batches, 1)
    return None


@wandb_init
@single_node_env.task
def single_node_auto(duration_seconds: int) -> typing.Optional[float]:
    """
    Single-node with run_mode="auto" (default).

    - Only rank 0 initializes W&B and logs
    - Other ranks get None from get_wandb_run()
    - Results in 1 W&B run
    """
    return _train_loop_impl(duration_seconds=duration_seconds)


@wandb_init(run_mode="shared")
@single_node_env.task
def single_node_shared(duration_seconds: int) -> typing.Optional[float]:
    """
    Single-node with run_mode="shared".

    - All ranks log to the same W&B run
    - Uses W&B shared mode with x_label to identify each rank
    - Results in 1 W&B run with metrics from all GPUs
    """
    return _train_loop_impl(duration_seconds=duration_seconds)


@wandb_init(run_mode="new")
@single_node_env.task
def single_node_new(duration_seconds: int) -> typing.Optional[float]:
    """
    Single-node with run_mode="new".

    - Each rank gets its own W&B run
    - Runs are grouped together in W&B UI
    - Results in N W&B runs (1 per GPU)
    - Run IDs: {run_name}-{action_name}-rank-{rank}
    """
    return _train_loop_impl(duration_seconds=duration_seconds)


@wandb_init
@multi_node_env.task
def multi_node_auto(duration_seconds: int) -> typing.Optional[float]:
    """
    Multi-node with run_mode="auto" (default).

    - Local rank 0 of each worker initializes W&B and logs
    - Other ranks get None from get_wandb_run()
    - Results in N W&B runs (1 per worker/node)
    - Run IDs: {run_name}-{action_name}-worker-{worker_index}
    """
    return _train_loop_impl(duration_seconds=duration_seconds)


@wandb_init(run_mode="shared")
@multi_node_env.task
def multi_node_shared(duration_seconds: int) -> typing.Optional[float]:
    """
    Multi-node with run_mode="shared".

    - All ranks within each worker log to a shared run
    - Each worker has its own shared W&B run
    - Results in N W&B runs (1 per worker/node)
    - Run IDs: {run_name}-{action_name}-worker-{worker_index}
    """
    return _train_loop_impl(duration_seconds=duration_seconds)


@wandb_init(run_mode="new")
@multi_node_env.task
def multi_node_new(duration_seconds: int) -> typing.Optional[float]:
    """
    Multi-node with run_mode="new".

    - Each rank gets its own W&B run
    - Runs are grouped per worker in W&B UI
    - Results in NxGPUs W&B runs
    - Run IDs: {run_name}-{action_name}-worker-{worker_index}-rank-{local_rank}
    """
    return _train_loop_impl(duration_seconds=duration_seconds)


if __name__ == "__main__":
    import sys

    flyte.init_from_config()

    scenario = sys.argv[1] if len(sys.argv) > 1 else "single_node_auto"

    scenarios = {
        # Single-node scenarios
        "single_node_auto": single_node_auto,
        "single_node_shared": single_node_shared,
        "single_node_new": single_node_new,
        # Multi-node scenarios
        "multi_node_auto": multi_node_auto,
        "multi_node_shared": multi_node_shared,
        "multi_node_new": multi_node_new,
    }

    if scenario not in scenarios:
        print(f"Unknown scenario: {scenario}")
        print(f"Available scenarios: {list(scenarios.keys())}")
        sys.exit(1)

    task_fn = scenarios[scenario]
    print(f"Running scenario: {scenario}")

    run = flyte.with_runcontext(
        custom_context=wandb_config(project="distributed-training-demo", entity="samhita-alla", tags=[scenario])
    ).run(task_fn, duration_seconds=300)
    print(f"Run URL: {run.url}")
