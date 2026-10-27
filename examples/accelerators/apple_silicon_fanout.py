"""
Fan out several gpu="AppleSilicon" tasks at once and measure how they share the Mac's GPU.

Every gpu="AppleSilicon" task runs as its own native process on the Mac, so a devbox can run several at once: there is
no queue in front of the GPU. They do share one GPU and its unified memory, though, so this example measures what a
fan-out actually buys. A CPU-only driver task starts `n` copies of the same small training job (a CNN on synthetic
MNIST-shaped data, so there's nothing to download and every task does the same work) and reports, per run:

- the makespan: from the first task starting its GPU work to the last one finishing,
- the aggregate throughput (images/s across all tasks) and the speedup over a single task,
- each task's peak memory, which is what runs out first on a busy Mac.

    flyte start devbox --apple-silicon
    flyte create config --devbox
    python examples/accelerators/apple_silicon_fanout.py            # sweeps n = 1, 2, 4, 8
    flyte run examples/accelerators/apple_silicon_fanout.py fanout --n 4

The task containers are only proxies for the processes on the Mac, so their CPU and memory requests are kept small:
they reserve devbox capacity but don't limit the work. Size the fan-out by the Mac's memory instead.
"""

import asyncio
import re
import subprocess
import threading
import time

import flyte

gpu_env = flyte.TaskEnvironment(
    name="apple-silicon-fanout-shard",
    image=flyte.Image.from_debian_base().with_pip_packages("torch"),
    # Small requests: the pod only proxies the process on the Mac, so these reserve devbox capacity, not GPU work.
    resources=flyte.Resources(cpu="250m", memory="512Mi", gpu="AppleSilicon"),
)

driver_env = flyte.TaskEnvironment(
    name="apple-silicon-fanout",
    image=flyte.Image.from_debian_base(),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    depends_on=[gpu_env],
)


def _gpu_utilization() -> int | None:
    """macOS's GPU utilization (%) from the IORegistry; None off macOS."""
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return None
    m = re.search(r'"Device Utilization %"=(\d+)', out)
    return int(m.group(1)) if m else None


@gpu_env.task
async def train_shard(shard: int, steps: int = 400, batch_size: int = 512) -> dict:
    """Train a small CNN on synthetic data on the Mac's GPU for a fixed number of steps; report timing and memory."""
    import os
    import resource

    import torch
    from torch import nn

    if not torch.backends.mps.is_available():
        raise RuntimeError("mps is not available: run with gpu='AppleSilicon' on an Apple Silicon devbox")
    device = torch.device("mps")
    torch.manual_seed(shard)

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    images = torch.randn(batch_size, 1, 28, 28, device=device)
    labels = torch.randint(0, 10, (batch_size,), device=device)

    def step() -> torch.Tensor:
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        return loss

    for _ in range(10):  # warm up: kernel compilation shouldn't count against the GPU work
        step()
    torch.mps.synchronize()

    samples: list[int] = []
    done = threading.Event()

    def sample() -> None:
        while not done.is_set():
            if (value := _gpu_utilization()) is not None:
                samples.append(value)
            done.wait(0.2)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    start = time.time()
    for _ in range(steps):
        loss = step()
    torch.mps.synchronize()
    end = time.time()
    done.set()
    sampler.join()

    seconds = end - start
    return {
        "shard": shard,
        "device": str(images.device),
        "pid": os.getpid(),
        "start": start,
        "end": end,
        "seconds": round(seconds, 2),
        "images_per_second": round(steps * batch_size / seconds),
        "final_loss": round(loss.item(), 4),
        # ru_maxrss is in bytes on macOS.
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6),
        "mps_driver_mb": round(torch.mps.driver_allocated_memory() / 1e6),
        "gpu_utilization_mean": round(sum(samples) / len(samples), 1) if samples else None,
    }


@driver_env.task
async def fanout(n: int = 4, steps: int = 400, batch_size: int = 512) -> dict:
    """Run `n` training shards on the Mac's GPU at once and summarize how they shared it."""
    submitted = time.time()
    shards = await asyncio.gather(*(train_shard(i, steps, batch_size) for i in range(n)))
    first_start = min(s["start"] for s in shards)
    last_end = max(s["end"] for s in shards)
    makespan = last_end - first_start
    return {
        "n": n,
        "distinct_host_processes": len({s["pid"] for s in shards}),
        # GPU work only: the first shard starting its steps to the last one finishing.
        "makespan_s": round(makespan, 2),
        # Everything: scheduling, pod start, the host process starting, importing torch, warm-up.
        "end_to_end_s": round(time.time() - submitted, 1),
        "mean_shard_s": round(sum(s["seconds"] for s in shards) / n, 2),
        # Did they actually run at the same time? The latest start before the earliest end means they all overlapped.
        "all_overlapped": max(s["start"] for s in shards) < min(s["end"] for s in shards),
        "aggregate_images_per_second": round(n * steps * batch_size / makespan),
        "max_peak_rss_mb": max(s["peak_rss_mb"] for s in shards),
        "shards": shards,
    }


if __name__ == "__main__":
    flyte.init_from_config()
    rows = []
    for n in (1, 2, 4, 8):
        run = flyte.run(fanout, n=n)
        print(run.url)
        run.wait()
        rows.append(run.outputs()[0])
    base = rows[0]["aggregate_images_per_second"]
    header = ("n", "makespan s", "mean shard s", "agg img/s", "speedup", "overlap", "peak MB")
    print(" ".join(f"{h:>{w}}" for h, w in zip(header, (3, 11, 13, 10, 8, 8, 8))))
    for r in rows:
        print(
            f"{r['n']:>3} {r['makespan_s']:>11} {r['mean_shard_s']:>13} {r['aggregate_images_per_second']:>10} "
            f"{r['aggregate_images_per_second'] / base:>7.2f}x {r['all_overlapped']!s:>8} {r['max_peak_rss_mb']:>8}"
        )
