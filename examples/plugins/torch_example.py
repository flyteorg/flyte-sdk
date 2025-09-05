# # Torch Example
import torch
import torch.nn as nn
import torch.optim as optim
from flyteplugins.pytorch.task import MasterNodeConfig, TorchJobConfig, WorkerNodeConfig
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import flyte
from flyte._context import internal_ctx

image = (
    flyte.Image.from_base("debian:slim")
    .clone(name="spark", python_version=(3, 10))
    .with_pip_packages("flyteplugins-spark", pre=True)
)


task_env = flyte.TaskEnvironment(
    name="task_env", resources=flyte.Resources(cpu=(1, 2), memory=("400Mi", "1000Mi")), image=image
)

torch_env = flyte.TaskEnvironment(
    name="torch_env",
    resources=flyte.Resources(cpu=(1, 2), memory=("400Mi", "1000Mi"), gpu=(1, 1)),
    plugin_config=TorchJobConfig(
        worker_node_config=WorkerNodeConfig(image="pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime", replicas=2),
        master_node_config=MasterNodeConfig(replicas=1),
    ),
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


@task_env.task
async def train_loop(model: nn.Module, rank: int, world_size: int, epochs: int = 200) -> float:
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
            print(f"Loss: {final_loss}")

    return final_loss


@torch_env.task
async def hello_torch_nested() -> float:
    model = LinearRegressionModel()
    ctx = internal_ctx()

    rank = ctx.data.task_context.data["rank"]
    world_size = ctx.data.task_context.data["world_size"]
    ddp_model = ctx.data.task_context.data["ddp_model"]

    return await train_loop(ddp_model(model), rank, world_size)


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(hello_torch_nested)
    print("run name:", run.name)
    print("run url:", run.url)
    run.wait(run)
