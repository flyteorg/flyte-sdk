# Flyte NVIDIA Nsight Systems plugin

Profile a Flyte task with NVIDIA Nsight Systems by adding one decorator. The task runs under
`nsys` automatically, its GPU metrics are summarized into the task's Flyte report, and the full
`.nsys-rep` trace comes back as a downloadable output. Your task's inputs, body, and return value
are unchanged, so profiling is something you switch on and off without rewriting the work.

## Install

```
pip install flyteplugins-nsight
```

The task image must have the `nsys` CLI on PATH. The NGC PyTorch images ship it, for example
`nvcr.io/nvidia/pytorch:24.08-py3`.

## Usage

```python
import flyte
from flyteplugins.nsight import nsys_profile

image = (
    flyte.Image.from_base("nvcr.io/nvidia/pytorch:24.08-py3")
    .clone(extendable=True, name="train", python_version=(3, 10))
    .with_pip_packages("flyte", "uv", "flyteplugins-nsight")
)

env = flyte.TaskEnvironment(
    name="train",
    image=image,
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L4:1"),
    # osrt tracing and GPU-counter sampling need CAP_SYS_ADMIN + unconfined AppArmor.
    pod_template=flyte.PodTemplate().allow_nested_sandboxing(),
)

@nsys_profile(trace=["cuda", "nvtx", "cudnn", "cublas"])
@env.task
async def train(epochs: int = 20) -> str:
    ...ordinary training code...
    return "done"
```

Running the task produces:

- a GPU Profile tab in the task's Flyte report, with summary tiles, a top-kernels chart, an NVTX
  range breakdown, and the full `nsys stats` tables, and
- the `.nsys-rep` trace as an output you can download and open in the Nsight Systems GUI (`nsys-ui`).

## Profiling only part of a task

Whole-task profiling of a long run produces an unwieldy trace. To profile just the hot loop, put
the task under nsys with `capture="manual"` and bracket the region:

```python
from flyteplugins.nsight import nsys, nvtx

@nsys_profile(capture="manual")
@env.task
async def train():
    warmup()
    async with nsys.range("hot-loop"):
        for step in range(100):
            with nvtx.range("step"):
                train_step()
```

`nsys.range` matches your task body. Use `async with` in an `async def` task, and plain `with` in a
`def` task, since `@nsys_profile` profiles both:

```python
@nsys_profile(capture="manual")
@env.task
def train():
    warmup()
    with nsys.range("hot-loop"):
        for step in range(100):
            with nvtx.range("step"):
                train_step()
```

Each region collects independently and produces its own report section and trace output.

## Labeling the timeline

Use `nvtx.range` to give regions of your code readable names on the timeline and in the NVTX
summary. It is a no-op when torch or CUDA is unavailable, so the same code runs off-GPU.

```python
from flyteplugins.nsight import nvtx

with nvtx.range("forward"):
    out = model(x)
```

## Options

- `trace`: nsys trace domains, for example cuda, nvtx, cudnn, cublas, osrt. osrt and GPU-counter
  sampling need elevated pod capabilities.
- `sample`: CPU sampling mode passed to `nsys -s`, for example "cpu" or "none". Omitted by default.
- `capture`: "task" profiles the whole body automatically. "manual" leaves collection to
  `nsys.range(...)` blocks.
- `reports`: which `nsys stats` reports to render.
- `attach_report`: also surface the `.nsys-rep` as a downloadable trace output. Defaults to True.
- `enabled`: when False the decorator is a transparent passthrough, so profiling can stay in the
  code and be turned off without removing it.

## Capabilities

`--trace=osrt` and GPU-counter sampling need `CAP_SYS_ADMIN` and an unconfined AppArmor profile, or
profiling silently yields an empty trace or an `ERR_NVGPUCTRPERM` error.
`flyte.PodTemplate().allow_nested_sandboxing()` grants that bundle. The cluster-clean alternative is
to set `NVreg_RestrictProfilingToAdminUsers=0` on the GPU nodes' driver. If you cannot grant either,
restrict `trace` to cuda,nvtx, which needs no extra capabilities.

## How it works

The decorator stamps the task's container with an instruction the Flyte runtime obeys to re-exec the
whole action under `nsys launch --session-new=...`. That establishes an interactive nsys session but
collects nothing yet. Inside the task the plugin runs `nsys start` before your code and `nsys stop`
after; `nsys stop` flushes the `.nsys-rep` to disk while the task keeps running, so the task can then
summarize it with `nsys stats` and return it through a traced function. None of this changes your
task's own signature.

Decorator order: `@nsys_profile` must be the outermost decorator, above `@env.task`.
