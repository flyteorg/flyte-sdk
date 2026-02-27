import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import flyte
from flyte.io import File

env = flyte.TaskEnvironment(name="mnist-training-example")


@env.task
async def load_mnist_data() -> File:
    """Download MNIST using torchvision and persist tensors for downstream tasks."""
    train_dataset = datasets.MNIST(
        root="/tmp/flyte_mnist",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_dataset = datasets.MNIST(
        root="/tmp/flyte_mnist",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    train_images = train_dataset.data.to(dtype=torch.float32) / 255.0
    train_labels = train_dataset.targets
    test_images = test_dataset.data.to(dtype=torch.float32) / 255.0
    test_labels = test_dataset.targets

    bundle = {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }
    local_fp = "/tmp/mnist_tensors.pt"
    torch.save(bundle, local_fp)
    return await File.from_local(local_fp)


@env.task
async def train_mnist_classifier(
    dataset_file: File,
    sample_size: int,
    test_size: float,
    num_epochs: int,
) -> dict[str, float]:
    """Train and evaluate a small PyTorch model on MNIST."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than 0.")

    torch.manual_seed(42)

    local_dataset_fp = await dataset_file.download()
    bundle = torch.load(local_dataset_fp, weights_only=False)
    train_images = bundle["train_images"].reshape(-1, 28 * 28)
    train_labels = bundle["train_labels"]
    default_test_images = bundle["test_images"].reshape(-1, 28 * 28)
    default_test_labels = bundle["test_labels"]

    sample_size = min(sample_size, len(train_images))
    sampled_images = train_images[:sample_size]
    sampled_labels = train_labels[:sample_size]

    split_idx = max(1, int(sample_size * (1.0 - test_size)))
    split_idx = min(split_idx, sample_size - 1) if sample_size > 1 else 1

    x_train_tensor = sampled_images[:split_idx]
    y_train_tensor = sampled_labels[:split_idx]
    x_test_tensor = sampled_images[split_idx:]
    y_test_tensor = sampled_labels[split_idx:]
    if len(x_test_tensor) == 0:
        x_test_tensor = default_test_images[:1000]
        y_test_tensor = default_test_labels[:1000]

    train_loader = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor),
        batch_size=64,
        shuffle=True,
    )

    model = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    model.train()
    for _ in range(num_epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(x_test_tensor).argmax(dim=1)
        accuracy = (predictions == y_test_tensor).float().mean().item()

    return {
        "accuracy": accuracy,
        "train_size": float(len(x_train_tensor)),
        "test_size": float(len(x_test_tensor)),
    }


@env.task
async def main(sample_size: int = 1000, test_size: float = 0.2, num_epochs: int = 1) -> dict[str, float]:
    dataset_file = await load_mnist_data()
    return await train_mnist_classifier(
        dataset_file=dataset_file,
        sample_size=sample_size,
        test_size=test_size,
        num_epochs=num_epochs,
    )
