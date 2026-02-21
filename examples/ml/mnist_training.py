"""MNIST training pipeline — demonstrates Flyte local caching, reports, and TUI."""

import io
import json
import base64

import flyte
import flyte.report
from flyte.io import File

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "torch", "torchvision", "matplotlib",
)

env = flyte.TaskEnvironment(
    name="ml_pipeline",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi", gpu=1),
)


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_model():
    """ResNet18 adapted for MNIST (1-channel 28x28 input)."""
    import torch.nn as nn
    from torchvision import models

    model = models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    model.maxpool = nn.Identity()
    return model


def fig_to_html(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}" />'


# NOTE: torch/torchvision are imported inside tasks (not at module level)
# to avoid file descriptor issues with Flyte's local subprocess management.


@env.task(cache="auto")
async def load_data(data_dir: str = "./data") -> str:
    """Download MNIST — cached after first run."""
    import os
    import urllib.request

    raw_dir = os.path.join(data_dir, "MNIST", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    for fname in files:
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(base_url + fname, path)

    print("Dataset ready.")
    return data_dir


@env.task
async def train(data_dir: str, epochs: int = 5, lr: float = 0.001, batch_size: int = 64) -> tuple[File, str]:
    """Train ResNet18 on MNIST, return model file and training history."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = get_device()
    print(f"Using device: {device}")
    model = create_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {
        "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "train_loss": [], "train_acc": [],
    }

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} — training...")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            correct += (output.argmax(1) == target).sum().item()
            total += data.size(0)
            batch_acc = correct / total
            history["train_loss"].append(loss.item())
            history["train_acc"].append(batch_acc)
            print(f"Batch {i + 1}/{len(train_loader)} — loss: {loss.item():.4f}, acc: {batch_acc:.4f}")

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}/{epochs} — loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}")

    path = "model.pt"
    torch.save(model.state_dict(), path)
    model_file = await File.from_local(path)

    return model_file, json.dumps(history)


@env.task
async def evaluate(model_file: File, data_dir: str) -> tuple[float, float]:
    """Evaluate model on test set, return accuracy and loss."""
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    local_path = await model_file.download()
    device = get_device()
    print(f"Using device: {device}")
    model = create_model().to(device)
    model.load_state_dict(torch.load(local_path, map_location=device, weights_only=True))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    correct = 0
    total = 0
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            correct += (output.argmax(1) == target).sum().item()
            total += data.size(0)

    test_acc = correct / total
    test_loss = test_loss / total
    print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
    return test_acc, test_loss


@env.task(report=True)
async def pipeline(epochs: int = 5, lr: float = 0.001, batch_size: int = 64, open_report: bool = False) -> tuple[str, File]:
    """Full MNIST pipeline — train, evaluate, and generate HTML report."""
    data_dir = await load_data()
    model_file, history_json = await train(data_dir, epochs=epochs, lr=lr, batch_size=batch_size)
    test_acc, test_loss = await evaluate(model_file, data_dir)

    # Build HTML report with training curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = json.loads(history_json)
    epoch_list = list(range(1, history["epochs"] + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epoch_list, history["train_loss"], "b-o", markersize=4)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    ax2.plot(epoch_list, history["train_acc"], "g-o", markersize=4)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Training Accuracy")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    charts_html = fig_to_html(fig)
    plt.close(fig)

    await flyte.report.replace.aio(
        f"<h2>MNIST Training Report</h2>"
        f"<h3>Hyperparameters</h3>"
        f"<table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;'>"
        f"<tr><td><b>Epochs</b></td><td>{history['epochs']}</td></tr>"
        f"<tr><td><b>Learning Rate</b></td><td>{history['lr']}</td></tr>"
        f"<tr><td><b>Batch Size</b></td><td>{history['batch_size']}</td></tr>"
        f"</table>"
        f"<h3>Training Curves</h3>{charts_html}"
        f"<h3>Test Results</h3>"
        f"<table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;'>"
        f"<tr><td><b>Test Accuracy</b></td><td>{test_acc:.4f}</td></tr>"
        f"<tr><td><b>Test Loss</b></td><td>{test_loss:.4f}</td></tr>"
        f"</table>"
    )
    await flyte.report.flush.aio()

    task_ctx = flyte.ctx()
    if task_ctx:
        from flyte._internal.runtime import io as flyte_io
        report_path = flyte_io.report_path(task_ctx.output_path)
        print(f"Report: {report_path}")
        if open_report:
            import webbrowser
            webbrowser.open(f"file://{report_path}")

    return f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}", model_file
