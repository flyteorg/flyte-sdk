"""
Flyte NVIDIA Nsight Systems plugin.

Run a Flyte task under Nsight Systems (`nsys`) by adding one decorator. The task runs under the
profiler automatically, its GPU metrics are summarized into the task's Flyte report, and the full
.nsys-rep trace is handed back as a downloadable output. Your task's inputs, body, and return value
are unchanged, so profiling is something you add and remove without touching the work itself.

How it works: the decorator stamps the task's container so the Flyte runtime re-execs the whole
action under `nsys launch`. Inside the task, collection is bracketed with `nsys start` / `nsys stop`
(`nsys stop` flushes the report to disk while the task keeps running), the trace is summarized with
`nsys stats`, and the .nsys-rep is returned through a traced function so it appears as a trace output.

Basic usage:

    import flyte
    from flyteplugins.nsight import nsys_profile

    env = flyte.TaskEnvironment(
        name="train",
        image=flyte.Image.from_base("nvcr.io/nvidia/pytorch:24.08-py3")
            .clone(extendable=True, name="train", python_version=(3, 10))
            .with_pip_packages("flyte", "uv"),
        resources=flyte.Resources(gpu="L4:1"),
        # osrt tracing and GPU counters need CAP_SYS_ADMIN + unconfined AppArmor.
        pod_template=flyte.PodTemplate().allow_nested_sandboxing(),
    )

    @nsys_profile(trace=["cuda", "nvtx", "cudnn", "cublas"])
    @env.task
    async def train(epochs: int = 20) -> str:
        ...ordinary training code...
        return "done"

Profile only a region of a long task:

    from flyteplugins.nsight import nsys, nvtx

    @nsys_profile(capture="manual")
    @env.task
    async def train():
        warmup()
        async with nsys.range("hot-loop"):
            for step in range(100):
                with nvtx.range("step"):
                    train_step()

Requirements:
- The task image must have the `nsys` CLI on PATH (NGC PyTorch images ship it).
- `trace` domains osrt and GPU-counter sampling need elevated pod capabilities; restrict to
  cuda,nvtx if you cannot grant them.

Decorator order: @nsys_profile must be the outermost decorator, above @env.task.
"""

from __future__ import annotations

from . import nsys, nvtx
from ._capture import capture_report_file
from ._control import nsys_available, session_name, under_nsys
from ._decorator import nsys_profile
from ._report import DEFAULT_REPORTS

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_REPORTS",
    "capture_report_file",
    "nsys",
    "nsys_available",
    "nsys_profile",
    "nvtx",
    "session_name",
    "under_nsys",
]
