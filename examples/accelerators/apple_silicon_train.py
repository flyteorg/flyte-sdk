"""
Train a model on the Mac's GPU from a devbox task: PyTorch on `mps` with `gpu="AppleSilicon"`.

A devbox container is a Linux VM with no Metal, so a `gpu="AppleSilicon"` task runs natively on the Mac instead, in an
environment built from the image's pip packages. Its code uses the GPU the way it would use CUDA on a cluster:
pick the device and move the model and data to it. Inputs, outputs and logs go through the devbox as usual.

This runs workload code outside any sandbox, so it is opt-in, and the devbox binds every port to 127.0.0.1:

    flyte start devbox --apple-silicon
    flyte create config --devbox
    python examples/accelerators/apple_silicon_train.py

The first run builds the Mac's environment (torch, torchvision) and downloads MNIST; later runs reuse both. The task's
report shows the loss curve as it trains.
"""

import time

import flyte
import flyte.io
import flyte.report

env = flyte.TaskEnvironment(
    name="apple-silicon-train",
    image=flyte.Image.from_debian_base().with_pip_packages("torch", "torchvision"),
    resources=flyte.Resources(cpu=2, memory="4Gi", gpu="AppleSilicon"),
)


CARD = "background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px 20px;margin:0 0 16px"
MUTED = "color:#6b7280;font-size:13px"


def _loss_chart(losses: list[float], epochs: int) -> str:
    """An inline SVG line chart of the training loss per epoch."""
    w, h, pad = 600, 200, 36
    top = max(losses) * 1.1
    xs = [pad + (w - 2 * pad) * (i / max(epochs - 1, 1)) for i in range(epochs)]
    ys = [h - pad - (h - 2 * pad) * (loss / top) for loss in losses]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#7652a2"/>' for x, y in zip(xs, ys))
    ticks = "".join(
        f'<text x="{x:.1f}" y="{h - 12}" font-size="11" text-anchor="middle" fill="#6b7280">{i + 1}</text>'
        for i, x in enumerate(xs)
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" style="width:100%;max-width:{w}px">'
        f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="#e5e7eb"/>'
        f'<text x="{pad}" y="16" font-size="11" fill="#6b7280">loss (axis max {top:.3f}) by epoch</text>'
        f'<polyline points="{line}" fill="none" stroke="#7652a2" stroke-width="2.5"/>{dots}{ticks}</svg>'
    )


def render(device: str, epochs: int, history: list[dict], metrics: dict | None = None) -> str:
    """The report: summary tiles, the loss curve, and a row per epoch. Re-rendered after every epoch."""
    status = "done" if metrics else f"training · epoch {len(history)}/{epochs}"
    tiles = [
        ("Device", device),
        ("Epochs", f"{len(history)}/{epochs}"),
        ("Last loss", f"{history[-1]['loss']:.4f}" if history else "-"),
        ("Throughput", f"{history[-1]['images_per_second']:,} img/s" if history else "-"),
    ]
    if metrics:
        tiles += [("Test accuracy", f"{metrics['test_accuracy']:.2%}"), ("Train time", f"{metrics['train_seconds']} s")]
    tile_html = "".join(
        f'<div style="{CARD};margin:0"><div style="{MUTED}">{k}</div>'
        f'<div style="font-size:22px;font-weight:700;margin-top:4px">{v}</div></div>'
        for k, v in tiles
    )
    th = 'style="text-align:left;padding:6px 8px;color:#6b7280;font-weight:500"'
    rows = "".join(
        f'<tr style="border-top:1px solid #f3f4f6"><td style="padding:6px 8px">{r["epoch"]}</td>'
        f'<td style="padding:6px 8px">{r["loss"]:.4f}</td><td style="padding:6px 8px">{r["seconds"]} s</td>'
        f'<td style="padding:6px 8px">{r["images_per_second"]:,}</td></tr>'
        for r in history
    )
    return (
        '<div style="font-family:system-ui,-apple-system,sans-serif;color:#111827;background:#f9fafb;'
        'padding:20px;max-width:960px">'
        f'<div style="{CARD}"><div style="font-size:20px;font-weight:700">MNIST CNN on the Mac\'s GPU</div>'
        f'<div style="{MUTED}">PyTorch on <code>{device}</code> · {status}</div></div>'
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;'
        f'margin-bottom:16px">{tile_html}</div>'
        f'<div style="{CARD}"><div style="font-weight:600;margin-bottom:8px">Training loss</div>'
        f"{_loss_chart([r['loss'] for r in history], epochs) if history else ''}</div>"
        f'<div style="{CARD}"><table style="width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums">'
        f"<tr><th {th}>Epoch</th><th {th}>Loss</th><th {th}>Time</th><th {th}>Images/s</th></tr>{rows}</table></div>"
        "</div>"
    )


@env.task(report=True)
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

    tab = flyte.report.get_tab("main")
    history: list[dict] = []
    tab.replace(render(str(device), epochs, history))
    await flyte.report.flush.aio()

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
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / len(train_set),
                "seconds": round(seconds, 1),
                "images_per_second": round(len(train_set) / seconds),
            }
        )
        tab.replace(render(str(device), epochs, history))
        await flyte.report.flush.aio()

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
    tab.replace(render(str(device), epochs, history, metrics))
    await flyte.report.flush.aio()
    return metrics, await flyte.io.File.from_local("mnist_cnn.pt")


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(train)
    print(run.url)
    run.wait()
    print(run.outputs())
