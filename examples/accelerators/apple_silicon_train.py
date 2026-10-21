"""
Train a model on the Mac's GPU from a devbox task: PyTorch on `mps` with `gpu="AppleSilicon"`.

A devbox container is a Linux VM with no Metal, so a `gpu="AppleSilicon"` task runs natively on the Mac instead, in an
environment built from the image's pip packages. Its code uses the GPU the way it would use CUDA on a cluster:
pick the device and move the model and data to it. Inputs, outputs and logs go through the devbox as usual.

This runs workload code outside any sandbox, so it is opt-in, and the devbox binds every port to 127.0.0.1:

    flyte start devbox --apple-silicon
    flyte create config --devbox
    python examples/accelerators/apple_silicon_train.py

The first run builds the Mac's environment (torch, torchvision) and downloads MNIST; later runs reuse both.
"""

import time

import flyte
import flyte.io

env = flyte.TaskEnvironment(
    name="apple-silicon-train",
    image=flyte.Image.from_debian_base().with_pip_packages("torch", "torchvision"),
    resources=flyte.Resources(cpu=2, memory="4Gi", gpu="AppleSilicon"),
)


@env.task
async def train(epochs: int = 3, batch_size: int = 256, lr: float = 1e-3) -> tuple[dict, flyte.io.File]:
    """Train a small CNN on MNIST on the Mac's GPU; return its metrics and weights."""
    import torch
    from torch import nn
    from torchvision import datasets, transforms

    # On a cluster this would be "cuda"; on the Mac's GPU it is "mps". Fail rather than train on the CPU.
    if not torch.backends.mps.is_available():
        raise RuntimeError("mps is not available: run with gpu='AppleSilicon' on an Apple Silicon devbox")
    device = torch.device("mps")

    to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST("data", train=True, download=True, transform=to_tensor)
    test_set = datasets.MNIST("data", train=False, download=True, transform=to_tensor)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1024)

    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        epoch_start, total_loss = time.perf_counter(), 0.0
        for batch, targets in train_loader:
            images, labels = batch.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        seconds = time.perf_counter() - epoch_start
        print(
            f"epoch {epoch + 1}/{epochs}: loss {total_loss / len(train_set):.4f}, "
            f"{len(train_set) / seconds:,.0f} images/s on {device}"
        )

    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            correct += (model(images.to(device)).argmax(1).cpu() == labels).sum().item()

    torch.save(model.state_dict(), "mnist_cnn.pt")
    metrics = {
        "device": str(next(model.parameters()).device),
        "epochs": epochs,
        "test_accuracy": round(correct / len(test_set), 4),
        "train_seconds": round(time.perf_counter() - start, 1),
    }
    return metrics, await flyte.io.File.from_local("mnist_cnn.pt")


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(train)
    print(run.url)
    run.wait()
    print(run.outputs())
