# Flyte DeepInfer Plugin

Run [DeepSeek's DeepInfer](https://github.com/deepseek-ai/DeepInfer), the
high-performance inference framework behind DeepSeek's production serving, as **Flyte
tasks** for offline/batch inference.

DeepInfer builds on vLLM and adds DeepSeek's optimised kernels (FlashMLA, DeepGEMM,
DeepEP), MoE expert-parallel load balancing (EPLB), speculative NextN decoding, and
structured outputs. It currently supports **DeepSeek-V4.1-Flash** on **NVIDIA SM100
GPUs** (B200 / GB200) and **Huawei Ascend 950 NPUs**.

## Installation

```bash
pip install --pre flyteplugins-deepinfer
```

## Quickstart

```python
import flyte
from flyteplugins.deepinfer import DeepInfer, deepinfer_image

env = flyte.TaskEnvironment(
    name="deepinfer",
    resources=flyte.Resources(gpu="B200:8"),
    image=deepinfer_image(),
)

llm = DeepInfer("deepseek-ai/DeepSeek-V4.1-Flash-1017")

@env.task
async def complete(prompts: list[str]) -> list[str]:
    return await llm.generate.aio(prompts, temperature=0, max_tokens=512)
```

That is the whole API surface for most tasks:

- **`DeepInfer(model, ...)`** is cheap to construct and touches no GPU. The first
  `generate` starts one engine process per GPU and waits for the model to load; the
  engines then stay up for the life of the task process.
- **`generate(prompts, ...)`** splits the prompts across the engines and returns one
  completion per prompt, **in input order**. Use `await llm.generate.aio(...)` in async
  tasks and `llm.generate(...)` in sync code.
- **`deepinfer_image()`** produces an image with DeepInfer built from source.

`model` can be a Hugging Face id, a local checkpoint directory, or a `flyte.io.Dir`
holding one (downloaded on first use).

## Generation options

```python
await llm.generate.aio(prompts, temperature=0.6, top_p=0.95, max_tokens=1024)
```

Options you leave unset fall back to DeepInfer's own `SamplingParams` defaults (note
upstream's default `max_tokens` is small). Any other `SamplingParams` field can be
passed as a keyword argument.

### Structured output

Constrain decoding with at most one of `json_schema`, `regex` or `grammar`:

```python
from pydantic import BaseModel

class Ticket(BaseModel):
    category: Literal["billing", "account", "bug", "other"]
    summary: str

tickets = await llm.generate.aio(prompts, json_schema=Ticket)            # list[Ticket]
labels = await llm.generate.aio(prompts, regex=r"(positive|negative)")   # list[str]
answers = await llm.generate.aio(prompts, grammar='root ::= "yes" | "no"')
```

A Pydantic model returns validated instances (which Flyte serializes as task outputs);
a dict or JSON-string schema returns parsed JSON.

## Warm engines on reusable containers

Loading DeepInfer (weights + CUDA warm-up) is far more expensive than a `generate` call.
Because `DeepInfer` keeps its engines running for the life of the process, a reusable
environment keeps them warm across task invocations; only the first call pays the load:

```python
env = flyte.TaskEnvironment(
    name="deepinfer",
    resources=flyte.Resources(gpu="B200:8"),
    image=deepinfer_image().with_pip_packages("unionai-reuse"),  # reuse needs this package
    reusable=flyte.ReusePolicy(replicas=(1, 2), idle_ttl=600),
)
llm = DeepInfer("deepseek-ai/DeepSeek-V4.1-Flash-1017")  # module level: shared by every call
```

To free the GPUs early, call `llm.close()` or use `with DeepInfer(...) as llm:`.

## Tuning the engines

Everything has a default; override only what you need:

```python
llm = DeepInfer(
    "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    cp_size=2,                                   # 2 GPUs per engine instead of 1
    env={"DEEPINFER_USE_NEXTN_DECODE": "0"},     # runtime flags, applied before import
    gpu_memory_utilization=0.85,                 # anything else goes to DeepInfer's LLM
    max_model_len=32768,
)
```

- `ep_size` (expert-parallel size) defaults to every device in the job, and the number
  of engines to `ep_size // cp_size`.
- The runtime flags default to DeepInfer's own example configuration (NextN decoding,
  EPLB, sampler top-k); `env=` overrides any of them.

## Images

```python
deepinfer_image()                                    # build DeepInfer from source
deepinfer_image(commit="<sha>")                      # ...at a different commit
deepinfer_image(base="registry/deepinfer:b200")      # or layer the plugin onto a prebuilt image
deepinfer_image(base="registry/deepinfer:b200", python="/opt/conda/bin/python")
```

- The source build starts from NVIDIA's official `nvidia/cuda:<version>-devel-ubuntu24.04`
  image, which already has the CUDA toolkit for both amd64 and arm64 (e.g. GB200 Grace
  hosts), so nothing CUDA-specific is installed by hand. DeepInfer has no tagged
  releases, so the build is **pinned to a commit**; pass `commit=` to move it.
- The build compiles DeepInfer's CUDA extensions and takes a while. Build it once and
  reuse the image, or have your platform team bake one and pass it as `base=`. If that
  image's DeepInfer lives in its own Python environment, point `python=` at it.
- The result is a regular `flyte.Image`: add anything else with `with_pip_packages`,
  `with_apt_packages`, and so on.

## Multi-node

DeepInfer's expert-parallel pool can span several nodes. `clustered_deepinfer_environment`
gives you a `ClusteredTaskEnvironment` (a Kubernetes JobSet) with one DeepInfer launcher
per node; the same `DeepInfer` code runs on every node, and `ep_size` defaults to every
GPU across all of them:

```python
cluster = clustered_deepinfer_environment(
    nodes=2, image=deepinfer_image(), resources=flyte.Resources(gpu="B200:8")  # per node
)
driver = flyte.TaskEnvironment(name="driver", image=..., depends_on=[cluster])
llm = DeepInfer("deepseek-ai/DeepSeek-V4.1-Flash-1017")

@cluster.task
async def generate_on_cluster(prompts: list[str], out_dir: str) -> None:
    results = await llm.generate.aio(prompts)  # this node's entries filled, the rest None
    ...  # write them under out_dir

@driver.task
async def multinode_generate(prompts: list[str]) -> list[str]:
    ...  # launch generate_on_cluster, then merge every node's results
```

Two Flyte rules shape multi-node code: only rank 0's return value becomes a clustered
task's output, and only a regular task can launch a clustered one. So each node writes
its results to shared storage and a regular driver task merges them. See
[`examples/deepinfer_multinode.py`](examples/deepinfer_multinode.py) for the full pattern.

## Examples

| Example | Shows |
| --- | --- |
| [`deepinfer_offline_inference.py`](examples/deepinfer_offline_inference.py) | The quickstart on a reusable container, so the engines stay warm between invocations. |
| [`deepinfer_structured_extraction.py`](examples/deepinfer_structured_extraction.py) | Pydantic-schema extraction and regex-constrained classification. |
| [`deepinfer_batch_pipeline.py`](examples/deepinfer_batch_pipeline.py) | Cache the checkpoint once in a `flyte.io.Dir`, fan shards of a JSONL `File` out over independent GPU nodes, write the results back to a `File`. |
| [`deepinfer_multinode.py`](examples/deepinfer_multinode.py) | One expert-parallel pool over several nodes, starting from a prebuilt image. |

## Limitations

- **Hardware-specific.** DeepInfer runs only on NVIDIA SM100 GPUs (B200 / GB200, CUDA
  13) and Huawei Ascend 950 NPUs (CANN 9.2); the engine cannot start without them.
- **Not a serving plugin.** This release of DeepInfer is offline/batch only; there is
  no OpenAI-compatible HTTP server.
- **Built from source.** No DeepInfer PyPI wheel exists; the image must contain a
  built `deepinfer` package (`deepinfer_image()` builds one).
