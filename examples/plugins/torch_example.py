import typing

import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim
from flyteplugins.pytorch.task import Elastic
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import flyte

image = flyte.Image.from_debian_base(name="torch").with_pip_packages("flyteplugins-pytorch", pre=True)

torch_env = flyte.TaskEnvironment(
    name="torch_env",
    resources=flyte.Resources(cpu=(1, 2), memory=("1Gi", "2Gi")),
    plugin_config=Elastic(
        nproc_per_node=1,
        # if you want to do local testing set nnodes=1
        nnodes=2,
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


def train_loop(epochs: int = 3) -> float:
    """
    A simple training loop for linear regression.
    """
    torch.distributed.init_process_group("gloo")
    model = DDP(LinearRegressionModel())

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    dataloader = prepare_dataloader(
        rank=rank,
        world_size=world_size,
        batch_size=64,
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
def torch_distributed_train(epochs: int) -> typing.Optional[float]:
    """
    A nested task that sets up a simple distributed training job using PyTorch's
    """
    print("starting launcher")
    loss = train_loop(epochs=epochs)
    print("Training complete")
    return loss


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote").run(torch_distributed_train, epochs=3)
    print("run name:", run.name)
    print("run url:", run.url)
