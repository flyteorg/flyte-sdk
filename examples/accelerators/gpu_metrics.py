# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte",
#    "torch==2.7.1",
# ]
# ///
"""
Exercise every GPU panel on the run details Metrics tab.

The task walks through distinct load regimes, each held long enough to be visible at the
console's polling resolution, so each chart has a recognizable signature:

  tensor      dense fp16 matmul          -> GPU util, SM active/occupancy, tensor-core activity,
                                            power draw, temperature, SM clock
  bandwidth   large elementwise sweeps    -> DRAM active, memory clock, framebuffer used
  pcie        pinned host<->device copies -> PCIe TX/RX
  nvlink      peer-to-peer device copies  -> NVLink bandwidth (needs 2+ GPUs on one node)
  idle        sleep                       -> everything drops, so regime edges are obvious

Pass trigger_xid=True to end the run with a deliberate out-of-bounds store from a Triton
kernel. That produces Xid 31 (GPU memory page fault) in the driver log, which DCGM reports
through DCGM_FI_DEV_XID_ERRORS and the console renders on the Xid strip. It kills the CUDA
context, so it always runs last; the task idles briefly, triggers, then lingers xid_linger_s
(default 45s) so DCGM scrapes the code while this pod still owns the GPU, and returns
successfully. After the fault the DCP profiling counters (SM active/occupancy, tensor, DRAM)
hold their last value until the process exits, so keep the linger short and use the default
duration so the phases before the fault are what dominates the charts.

after_xid decides how the attempt ends once the fault has been scraped, which is what
selects between the three ways a GPU fault reaches the console:

  return   finish normally             -> the fault shows on the charts only
  fail     raise a RuntimeError        -> a task failure with the fault alongside it,
                                          the shape a workload takes when its next CUDA
                                          call raises after the context has died
  kill     exit the process with 137   -> a pod-level failure with no error record, the
                                          shape a hardware fault usually takes

    flyte run examples/accelerators/gpu_metrics.py main --trigger_xid --after_xid fail

Pick the accelerator with GPU_METRICS_DEVICE. The default, T4:1, drives every panel except
NVLink (a single T4 has no NVLink, so the console hides that panel). A bare number requests
that many GPUs with no device pin, which is the way to reach a multi-GPU node whose
accelerator label the device map does not know:

    flyte run examples/accelerators/gpu_metrics.py main --duration_s 480
    GPU_METRICS_DEVICE=2 flyte run examples/accelerators/gpu_metrics.py main --duration_s 600

Throttling and memory-error panels need the matching dcgm-exporter fields enabled on the
cluster; on a T4, remapped rows is absent (Ampere and newer only).
"""

import os
import subprocess
import time
from typing import Any

import flyte

# Triton ships with torch on Linux but not on macOS, and the script is also imported locally by
# `flyte run`, so keep it optional. The kernel has to be defined at module scope: Triton's JIT
# resolves names in the kernel body and its annotations (`tl.constexpr`) from the function's
# globals, so a `tl` imported inside a helper is invisible to it.
try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - local import on macOS
    triton = None
    tl = None

if triton is not None:

    @triton.jit
    def _oob_store(ptr, stride, BLOCK: tl.constexpr):
        # stride of 2^28 floats = 1 GiB per lane, so every lane past the first lands far outside
        # the 16-float buffer this is launched with: an illegal address, i.e. Xid 31.
        # int64 offsets: 1023 lanes x 2^28 overflows int32 and would wrap back into range.
        offs = tl.arange(0, BLOCK).to(tl.int64) * stride
        tl.store(ptr + offs, tl.zeros([BLOCK], dtype=tl.float32))


# What the task does once the Xid has been scraped. "return" finishes normally, "fail"
# raises, and "kill" exits the process outright. See the module docstring.
_AFTER_XID_MODES = ("return", "fail", "kill")

_device_env = os.environ.get("GPU_METRICS_DEVICE") or "T4:1"
# "2" -> two GPUs of any kind (no accelerator selector); "T4:1" -> the pinned device.
DEVICE: str | int = int(_device_env) if _device_env.isdigit() else _device_env

image = flyte.Image.from_uv_script(__file__, name="gpu-metrics")

env = flyte.TaskEnvironment(
    name="gpu_metrics",
    resources=flyte.Resources(cpu=2, memory="10Gi", gpu=DEVICE, shm="auto"),  # type: ignore[arg-type]
    image=image,
)


def _log(msg: str) -> None:
    print(f"[gpu-metrics {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _nvidia_smi() -> None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        _log("nvidia-smi:\n" + out.stdout.strip())
    except (OSError, subprocess.SubprocessError) as e:  # nvidia-smi absent inside some images
        _log(f"nvidia-smi unavailable: {e}")


def _phase_tensor(torch: Any, dev: Any, seconds: float, n: int = 8192) -> float:
    """Dense fp16 matmul, the tensor-core hot loop. Returns achieved TFLOP/s."""
    a = torch.randn(n, n, device=dev, dtype=torch.float16)
    b = torch.randn(n, n, device=dev, dtype=torch.float16)
    flops_per = 2.0 * n * n * n
    iters, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(8):
            a = a @ b
            a.mul_(1e-3)  # keep values from overflowing fp16
        torch.cuda.synchronize(dev)
        iters += 8
    return iters * flops_per / (time.perf_counter() - t0) / 1e12


def _phase_bandwidth(torch: Any, dev: Any, seconds: float, gib: float = 4.0) -> float:
    """Elementwise sweeps over a large buffer, memory-bound. Returns GB/s of traffic."""
    n = int(gib * (1 << 30) / 4)
    x = torch.randn(n, device=dev, dtype=torch.float32)
    y = torch.empty_like(x)
    moved, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(4):
            torch.add(x, 1.0, out=y)  # read x, write y
            torch.mul(y, 0.5, out=x)  # read y, write x
        torch.cuda.synchronize(dev)
        moved += 4 * 2 * 2 * x.numel() * 4
    return moved / (time.perf_counter() - t0) / 1e9


def _phase_pcie(torch: Any, dev: Any, seconds: float, mib: int = 512) -> float:
    """Pinned host<->device round trips. Returns GB/s across the bus (both directions)."""
    host = torch.empty(mib << 20, dtype=torch.uint8).pin_memory()
    device = torch.empty(mib << 20, dtype=torch.uint8, device=dev)
    moved, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        device.copy_(host, non_blocking=True)
        host.copy_(device, non_blocking=True)
        torch.cuda.synchronize(dev)
        moved += 2 * host.numel()
    return moved / (time.perf_counter() - t0) / 1e9


def _phase_nvlink(torch: Any, seconds: float, gib: float = 1.0) -> float | None:
    """Peer-to-peer copies between GPU 0 and GPU 1. Returns GB/s, or None with one GPU."""
    if torch.cuda.device_count() < 2:
        _log("nvlink phase skipped: fewer than 2 GPUs visible")
        return None
    if not torch.cuda.can_device_access_peer(0, 1):
        _log("nvlink phase: peer access not available between GPU 0 and 1; copies go via host")
    n = int(gib * (1 << 30))
    src = torch.empty(n, dtype=torch.uint8, device="cuda:0")
    dst = torch.empty(n, dtype=torch.uint8, device="cuda:1")
    moved, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        dst.copy_(src, non_blocking=True)
        src.copy_(dst, non_blocking=True)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        moved += 2 * n
    return moved / (time.perf_counter() - t0) / 1e9


def _trigger_xid31(torch: Any) -> str:
    """Out-of-bounds store from the module-level Triton kernel: an illegal address, i.e. Xid 31."""
    if triton is None:
        return "triton not importable in this image; Xid not triggered"

    buf = torch.zeros(16, device="cuda:0", dtype=torch.float32)
    _log("triggering Xid 31 with a deliberate out-of-bounds store; the CUDA context will die")
    try:
        _oob_store[(1,)](buf, 1 << 28, BLOCK=1024)
        torch.cuda.synchronize(0)
    except RuntimeError as e:  # "an illegal memory access was encountered"
        return f"Xid 31 triggered: {str(e).splitlines()[0]}"
    except Exception as e:  # compile-time surprises should not hide the rest of the run's results
        return f"Xid trigger failed before launch: {type(e).__name__}: {e}"
    return "kernel completed without a fault (unexpected); Xid not triggered"


@env.task
def main(
    duration_s: int = 600,
    phase_s: int = 60,
    idle_s: int = 20,
    trigger_xid: bool = False,
    xid_linger_s: int = 45,
    after_xid: str = "return",
) -> dict[str, Any]:
    import torch

    if after_xid not in _AFTER_XID_MODES:
        raise ValueError(f"after_xid must be one of {sorted(_AFTER_XID_MODES)}, got {after_xid!r}")

    _nvidia_smi()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this task; check the accelerator request")

    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(dev)
    _log(f"{torch.cuda.device_count()} GPU(s); GPU 0 = {props.name}, {props.total_memory / 2**30:.0f} GiB")

    results: dict[str, Any] = {"device": props.name, "gpu_count": torch.cuda.device_count(), "phases": []}
    t_end = time.time() + duration_s
    cycle = 0
    while time.time() < t_end:
        cycle += 1
        for name, fn in (
            ("tensor", lambda: _phase_tensor(torch, dev, phase_s)),
            ("bandwidth", lambda: _phase_bandwidth(torch, dev, phase_s)),
            ("pcie", lambda: _phase_pcie(torch, dev, phase_s)),
            ("nvlink", lambda: _phase_nvlink(torch, phase_s)),
        ):
            if time.time() >= t_end:
                break
            _log(f"cycle {cycle}: {name} for {phase_s}s")
            value = fn()
            _log(f"cycle {cycle}: {name} -> {value}")
            results["phases"].append({"cycle": cycle, "phase": name, "value": value})
            _log(f"idle for {idle_s}s")
            time.sleep(idle_s)

    if trigger_xid:
        # Let the GPU go quiet first, so the last real samples before the fault are a clean
        # idle baseline rather than the tail of a busy phase.
        _log(f"idle for {idle_s}s before triggering the Xid")
        time.sleep(idle_s)
        results["xid"] = _trigger_xid31(torch)
        _log(results["xid"])
        # DCGM only attributes a GPU's samples to this pod while the pod is Running and holds
        # the device, and the Xid gauge is scraped every 15-30s. Returning right away would let
        # the pod exit before the scrape that carries the code, so stay alive for a few scrapes.
        # Keep this short: once the CUDA context is dead the DCP profiling counters (SM active,
        # SM occupancy, tensor, DRAM) hold their last value until the process exits, so a long
        # linger reads as a flat line on those charts.
        _log(f"lingering {xid_linger_s}s so DCGM scrapes the Xid while this pod still owns the GPU")
        time.sleep(xid_linger_s)
        if after_xid == "kill":
            # End the attempt the way a hardware fault usually does: the pod dies without
            # writing an error record, so the platform sees a pod-level failure rather than
            # a task that reported its own error. This cannot be a signal. The task is PID 1
            # of its container, and the kernel drops signals sent to PID 1 from inside its
            # own namespace, SIGKILL included, so os.kill would return without doing
            # anything. Exit status 137 (128 + SIGKILL) is what the kubelet records for a
            # killed container, and exiting this way skips the runtime's error handling
            # just as a real kill would.
            _log("exiting with status 137 so the pod fails at the pod level")
            os._exit(137)
        if after_xid == "fail":
            # End the attempt as an ordinary task failure, the shape a workload takes when
            # its next CUDA call raises after the context has died.
            raise RuntimeError("failing after the deliberate Xid")

    return results


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, duration_s=480, trigger_xid=True)
    print(r.url)
