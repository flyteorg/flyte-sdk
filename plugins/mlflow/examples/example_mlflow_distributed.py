"""
This example demonstrates distributed training with MLflow logging.

In distributed training, only rank 0 should log to MLflow to avoid
duplicate entries and conflicts. The @mlflow_run decorator auto-detects
the RANK environment variable and skips MLflow operations on non-zero ranks.
"""

import os
import time
import typing
from pathlib import Path

import flyte
import mlflow
import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim
from flyte._image import PythonWheels
from flyteplugins.pytorch.task import Elastic
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from flyteplugins.mlflow import Mlflow, get_mlflow_run, mlflow_config, mlflow_run

DATABRICKS_USERNAME = "<username>"
DATABRICKS_HOST = "<host>"


image = (
    flyte.Image.from_debian_base(name="torch-mlflow")
    .clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-mlflow",
        ),
    )
    .with_pip_packages(
        "flyteplugins-pytorch",
        "mlflow[databricks]",
    )
)

# Single-node environment (1 node, 4 GPUs)
single_node_env = flyte.TaskEnvironment(
    name="single_node_env",
    resources=flyte.Resources(cpu=(1, 2), memory=("1Gi", "10Gi"), gpu="T4:4", shm="auto"),
    plugin_config=Elastic(
        nproc_per_node=4,
        nnodes=1,
    ),
    image=image,
    secrets=[flyte.Secret(key="databricks_token", as_env_var="DATABRICKS_TOKEN")],
    env_vars={
        "NCCL_DEBUG": "INFO",
        "TORCH_NCCL_TRACE_BUFFER_SIZE": "1000",
        "MLFLOW_TRACKING_URI": "databricks",
        "GIT_PYTHON_REFRESH": "quiet",
        "DATABRICKS_HOST": DATABRICKS_HOST,
    },
)

# Multi-node environment (2 nodes, 4 GPUs each)
multi_node_env = flyte.TaskEnvironment(
    name="multi_node_env",
    resources=flyte.Resources(cpu=(1, 2), memory=("1Gi", "10Gi"), gpu="T4:4", shm="auto"),
    plugin_config=Elastic(
        nproc_per_node=4,
        nnodes=2,
    ),
    image=image,
    secrets=[flyte.Secret(key="databricks_token", as_env_var="DATABRICKS_TOKEN")],
    env_vars={
        "NCCL_DEBUG": "INFO",
        "TORCH_NCCL_TRACE_BUFFER_SIZE": "1000",
        "MLFLOW_TRACKING_URI": "databricks",
        "GIT_PYTHON_REFRESH": "quiet",
        "DATABRICKS_HOST": DATABRICKS_HOST,
    },
)


class MLP(nn.Module):
    """Large MLP to drive meaningful GPU utilization."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 4096, num_classes: int = 10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
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


def _train_loop_impl(num_epochs: int = 10) -> float | None:
    """Core training loop. Runs for a set number of epochs."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    torch.distributed.init_process_group("nccl", device_id=device)

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Model
    model = MLP(input_dim=784, hidden_dim=4096, num_classes=10).to(device)
    model = DDP(model, device_ids=[local_rank])

    # Dataset and dataloader
    dataset = SyntheticDataset(num_samples=50000, input_dim=784, num_classes=10)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=1024,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader))

    # MLflow - only rank 0 gets a run (others get None)
    mlflow_run_obj = get_mlflow_run()
    print(f"[Rank {rank}] mlflow_run: {mlflow_run_obj}")

    start_time = time.time()
    global_step = 0
    running_loss = 0.0
    num_batches = 0

    print(
        f"[Rank {rank}] Starting training for {num_epochs} epochs | "
        f"Dataset: {len(dataset)} samples | Batch size: 128 | Workers: {world_size}"
    )

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        model.train()

        for d, t in dataloader:
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

            # Log every 100 steps (only rank 0 has mlflow_run_obj)
            if global_step % 100 == 0:
                avg_loss = running_loss / num_batches
                elapsed = time.time() - start_time
                throughput = global_step * 1024 * world_size / elapsed

                if mlflow_run_obj:
                    mlflow.log_metrics(
                        {
                            "train/loss": avg_loss,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                            "train/global_step": global_step,
                            "train/throughput_samples_per_sec": throughput,
                        },
                        step=global_step,
                    )

                if local_rank == 0:
                    print(
                        f"[Epoch {epoch + 1}/{num_epochs} Step {global_step}] "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"Throughput: {throughput:.0f} samples/s"
                    )

                running_loss = 0.0
                num_batches = 0

    total_time = time.time() - start_time
    final_loss = running_loss / max(num_batches, 1)

    if local_rank == 0:
        print(f"Training complete | Epochs: {num_epochs} | Steps: {global_step} | Time: {total_time:.1f}s")

    # Log final metrics
    if mlflow_run_obj:
        mlflow.log_metrics(
            {
                "train/total_epochs": num_epochs,
                "train/total_steps": global_step,
                "train/total_time_seconds": total_time,
                "train/final_loss": final_loss,
            }
        )

    # Barrier to ensure all ranks finish training before cleanup
    print(f"[Rank {rank}] Waiting at barrier...")
    torch.distributed.barrier()
    print(f"[Rank {rank}] Barrier passed, calling destroy_process_group...")
    torch.distributed.destroy_process_group()
    print(f"[Rank {rank}] destroy_process_group done")

    # Each worker's primary (local_rank 0) returns a value
    if local_rank == 0:
        return final_loss
    return None


@mlflow_run
@single_node_env.task(links=(Mlflow()))
def single_node_auto(num_epochs: int) -> typing.Optional[float]:
    """
    Single-node with run_mode="auto" (default).

    - Only rank 0 initializes MLflow and logs
    - Other ranks get None from get_mlflow_run()
    - Results in 1 MLflow run
    """
    return _train_loop_impl(num_epochs=num_epochs)


@mlflow_run(run_mode="new")
@single_node_env.task(
    links=(Mlflow())
)  # since the run ID cannot be determined ahead of time, there will be no link in the UI
def single_node_new(num_epochs: int) -> typing.Optional[float]:
    """
    Single-node with run_mode="new".

    - Only rank 0 initializes MLflow and logs
    - Creates a new MLflow run (useful when called from a parent task)
    - Results in 1 MLflow run
    """
    return _train_loop_impl(num_epochs=num_epochs)


@mlflow_run
@multi_node_env.task(links=(Mlflow()))
def multi_node_auto(num_epochs: int) -> typing.Optional[float]:
    """
    Multi-node with run_mode="auto" (default).

    - Only global rank 0 initializes MLflow and logs
    - Other ranks get None from get_mlflow_run()
    - Results in 1 MLflow run total
    - Run ID: {run_name}-{action_name}
    """
    return _train_loop_impl(num_epochs=num_epochs)


@mlflow_run(run_mode="new")
@multi_node_env.task(
    links=(Mlflow())
)  # since the run ID cannot be determined ahead of time, there will be no link in the UI
def multi_node_new(num_epochs: int) -> typing.Optional[float]:
    """
    Multi-node with run_mode="new".

    - Only global rank 0 initializes MLflow and logs
    - Creates a new MLflow run (useful when called from a parent task)
    - Results in 1 MLflow run total
    - Run ID: {run_name}-{action_name}
    """
    return _train_loop_impl(num_epochs=num_epochs)


# Parent task that initializes the MLflow run and calls a distributed
# training child task. The link auto-propagates via link_host in mlflow_config().
parent_env = flyte.TaskEnvironment(
    name="parent_env",
    image=image,
    secrets=flyte.Secret(key="databricks_token", as_env_var="DATABRICKS_TOKEN"),
    env_vars={
        "MLFLOW_TRACKING_URI": "databricks",
        "GIT_PYTHON_REFRESH": "quiet",
        "DATABRICKS_HOST": DATABRICKS_HOST,
    },
    depends_on=[single_node_env, multi_node_env],
)


@mlflow_run
@parent_env.task
async def run_distributed_training(scenario: str, num_epochs: int = 10) -> typing.Optional[float]:
    scenarios = {
        "single_node_auto": single_node_auto,
        "single_node_new": single_node_new,
        "multi_node_auto": multi_node_auto,
        "multi_node_new": multi_node_new,
    }

    task_fn = scenarios[scenario]
    return task_fn(num_epochs=num_epochs)


if __name__ == "__main__":
    import sys

    flyte.init_from_config()

    scenario = sys.argv[1] if len(sys.argv) > 1 else "single_node_auto"
    print(f"Running scenario: {scenario}")

    run = flyte.with_runcontext(
        custom_context=mlflow_config(
            experiment_name=f"/Users/{DATABRICKS_USERNAME}/distributed-training-demo",
            tags={"scenario": scenario},
            link_host=DATABRICKS_HOST,
            link_template="{host}/ml/experiments/{experiment_id}/runs/{run_id}",
        )
    ).run(run_distributed_training, scenario=scenario, num_epochs=10)
    print(f"Run URL: {run.url}")
