# # Torch Example
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from flyteplugins.pytorch.task import MasterNodeConfig, TorchJobConfig, WorkerNodeConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import flyte
from flyte._context import internal_ctx

image = (
    flyte.Image.from_base("pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime")
    .clone(name="torch", python_version=(3, 10))
    # .with_pip_packages("flyteplugins-pytorch", pre=True)
    .with_pip_packages("torch==2.8.0", "numpy", "pandas")
    .with_source_folder(Path(__file__).parent.parent.parent / "plugins/pytorch", "./pytorch")
    .with_env_vars({"PYTHONPATH": "./pytorch/src:${PYTHONPATH}"})
    .with_local_v2()
)

torch_config = TorchJobConfig(
    worker_node_config=WorkerNodeConfig(image="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime", replicas=2),
    master_node_config=MasterNodeConfig(replicas=1),
    nproc_per_node=2,
    nnodes=2,
)


task_env = flyte.TaskEnvironment(
    name="task_env", resources=flyte.Resources(cpu=(1, 2), memory=("400Mi", "1000Mi")), image=image
)

torch_env = flyte.TaskEnvironment(
    name="torch_env",
    resources=flyte.Resources(cpu=(1, 2), memory=("1Gi", "2Gi")),
    plugin_config=torch_config,
    image=image,
)


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def prepare_dataloader(rank: int, world_size: int, batch_size: int = 2) -> DataLoader:
    """
    Prepare a DataLoader with a DistributedSampler so each rank
    gets a shard of the dataset.
    """
    # Dummy dataset
    x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y_train = torch.tensor([[3.0], [5.0], [7.0], [9.0]])
    dataset = TensorDataset(x_train, y_train)

    # Distributed-aware sampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def train_loop(epochs: int = 200) -> float:
    """
    A simple training loop for linear regression.
    """
    model = DDP(LinearRegressionModel())

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    dataloader = prepare_dataloader(
        rank=rank,
        world_size=world_size,
        batch_size=2,
    )

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    final_loss = 0.0

    for _ in range(epochs):
        for x, y in dataloader:
            outputs = model(x)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_loss = loss.item()
        if torch.distributed.get_rank() == 0:
            print(f"Loss: {final_loss}")

    return final_loss


@torch_env.task
def hello_torch_nested() -> None:
    """
    A nested task that sets up a simple distributed training job using PyTorch's
    """
    ctx = internal_ctx()
    launcher = ctx.data.task_context.data["elastic_launcher"]
    print("starting launcher")
    out = launcher(train_loop)
    print("Training complete")
    print("Final loss:", out)


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(hello_torch_nested)
    print("run name:", run.name)
    print("run url:", run.url)
