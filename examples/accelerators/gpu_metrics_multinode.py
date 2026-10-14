"""
Multi-node variant of gpu_metrics.py: exercise the GPU panels on the run details Metrics tab
with a ClusteredTaskEnvironment (JobSet + torchrun), one pod per node.

Each rank runs the same load regimes as the single-node example (tensor, bandwidth, PCIe), and
the ranks additionally run an all-reduce phase so the cross-node interconnect is exercised: on
T4 nodes without NVLink or RDMA that traffic goes NCCL -> TCP, which shows up as PCIe traffic
(host staging) and utilization on every GPU at once. In the console the action has one pod per
replica; every per-GPU chart shows one line per pod (pod chips select one), and CPU/Memory
charts follow the same pods.

Sized for dogfood's T4 pool by default: g4dn.xlarge nodes carry one T4 each, so REPLICAS=2 and
NPROC_PER_NODE=1 give a two-node, two-GPU world. Override with GPU_METRICS_DEVICE / REPLICAS /
NPROC_PER_NODE (NPROC_PER_NODE must not exceed the GPUs per device).

The image is built from the local flyte wheels (the container needs the `clustered` runtime
entrypoint), like examples/clustered/ddp_train_gpu.py. Build them once, then run:

    uv build --wheel
    uv run python examples/accelerators/gpu_metrics_multinode.py
"""

from __future__ import annotations

import os
import time
from typing import Any

import flyte
from flyte._image import DIST_FOLDER, PythonWheels
from flyte.clustered import ClusteredTaskEnvironment, ClusterFailurePolicy, TorchRun

_device_env = os.environ.get("GPU_METRICS_DEVICE") or "T4:1"
DEVICE: str | int = int(_device_env) if _device_env.isdigit() else _device_env
REPLICAS = int(os.environ.get("REPLICAS", "2"))
NPROC_PER_NODE = int(os.environ.get("NPROC_PER_NODE", "1"))

image = (
    flyte.Image.from_debian_base(name="gpu-metrics-multinode")
    .clone(addl_layer=PythonWheels(wheel_dir=DIST_FOLDER, package_name="flyte"))
    .with_pip_packages("torch==2.7.1")
)

env = ClusteredTaskEnvironment(
    name="gpu_metrics_multinode",
    image=image,
    resources=flyte.Resources(cpu=2, memory="10Gi", gpu=DEVICE, shm="auto"),  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    replicas=REPLICAS,
    nproc_per_node=NPROC_PER_NODE,
    runtime=TorchRun(rdzv_backend="static", max_restarts=0),
    failure_policy=ClusterFailurePolicy(max_restarts=0),
)


def _log(rank: int, msg: str) -> None:
    print(f"[gpu-metrics rank {rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _phase_tensor(torch: Any, dev: Any, seconds: float, n: int = 8192) -> float:
    a = torch.randn(n, n, device=dev, dtype=torch.float16)
    b = torch.randn(n, n, device=dev, dtype=torch.float16)
    iters, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(8):
            a = a @ b
            a.mul_(1e-3)
        torch.cuda.synchronize(dev)
        iters += 8
    return iters * 2.0 * n * n * n / (time.perf_counter() - t0) / 1e12


def _phase_bandwidth(torch: Any, dev: Any, seconds: float, gib: float = 4.0) -> float:
    n = int(gib * (1 << 30) / 4)
    x = torch.randn(n, device=dev, dtype=torch.float32)
    y = torch.empty_like(x)
    moved, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(4):
            torch.add(x, 1.0, out=y)
            torch.mul(y, 0.5, out=x)
        torch.cuda.synchronize(dev)
        moved += 4 * 2 * 2 * x.numel() * 4
    return moved / (time.perf_counter() - t0) / 1e9


def _phase_pcie(torch: Any, dev: Any, seconds: float, mib: int = 512) -> float:
    host = torch.empty(mib << 20, dtype=torch.uint8).pin_memory()
    device = torch.empty(mib << 20, dtype=torch.uint8, device=dev)
    moved, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        device.copy_(host, non_blocking=True)
        host.copy_(device, non_blocking=True)
        torch.cuda.synchronize(dev)
        moved += 2 * host.numel()
    return moved / (time.perf_counter() - t0) / 1e9


def _phase_allreduce(torch: Any, dist: Any, dev: Any, seconds: float, mib: int = 256) -> float:
    """NCCL all-reduce across the whole world. Returns algorithmic GB/s per rank."""
    buf = torch.ones(mib << 18, device=dev, dtype=torch.float32)  # mib MiB of fp32
    moved, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        dist.all_reduce(buf)
        torch.cuda.synchronize(dev)
        moved += buf.numel() * 4
    return moved / (time.perf_counter() - t0) / 1e9


@env.task
def main(duration_s: int = 600, phase_s: int = 60, idle_s: int = 20) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    ctx = flyte.ctx()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this task; check the accelerator request")

    # torchrun populated RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT. Pin this rank to its local
    # GPU before init_process_group so NCCL binds to the right device.
    local_rank = ctx.local_rank or 0
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    props = torch.cuda.get_device_properties(dev)
    _log(rank, f"world={world} node_rank={ctx.node_rank} nnodes={ctx.nnodes} device={props.name}")

    results: dict[str, Any] = {"rank": rank, "world_size": world, "device": props.name, "phases": []}
    t_end = time.time() + duration_s
    cycle = 0
    while time.time() < t_end:
        cycle += 1
        for name, fn in (
            ("tensor", lambda: _phase_tensor(torch, dev, phase_s)),
            ("bandwidth", lambda: _phase_bandwidth(torch, dev, phase_s)),
            ("pcie", lambda: _phase_pcie(torch, dev, phase_s)),
            ("allreduce", lambda: _phase_allreduce(torch, dist, dev, phase_s)),
        ):
            if time.time() >= t_end:
                break
            # Keep the ranks in step so each phase starts on every GPU at the same time and the
            # per-pod lines on a chart rise and fall together.
            dist.barrier()
            _log(rank, f"cycle {cycle}: {name} for {phase_s}s")
            value = fn()
            _log(rank, f"cycle {cycle}: {name} -> {value}")
            results["phases"].append({"cycle": cycle, "phase": name, "value": value})
            time.sleep(idle_s)

    dist.barrier()
    dist.destroy_process_group()
    return results


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, duration_s=480)
    print(r.url)
