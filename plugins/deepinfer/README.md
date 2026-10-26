# Flyte DeepInfer Plugin

Run [DeepSeek's DeepInfer](https://github.com/deepseek-ai/DeepInfer) — the
high-performance, large-scale inference framework behind DeepSeek's production
serving — as **Flyte tasks** for offline/batch inference.

DeepInfer is built primarily on vLLM and adds DeepSeek's optimised kernels
(FlashMLA, DeepGEMM, DeepEP, TileKernels), MoE expert-parallel load balancing
(EPLB), speculative & confidence-scheduled verification (DSpark / nextn), SWA
bounded replay, and structured outputs. It currently supports **DeepSeek-V4.1-Flash**
on **NVIDIA SM100 GPUs** and **Huawei Ascend 950 NPUs**.

This plugin does not re-serve an HTTP endpoint (DeepInfer's open-source release is
offline/batch only). Instead it gives you the **task-side plumbing** for running
DeepInfer correctly inside a `flyte.TaskEnvironment`, which is easy to get subtly
wrong:

- **Environment ordering** — DeepInfer only honours `CUDA_VISIBLE_DEVICES` and its
  runtime flags if they are set *before* `import deepinfer` runs. `create_llm` hides
  that prologue behind one call.
- **Model resolution** — accept an HF model id, a local checkpoint, or a mounted
  Flyte `Dir`/`RunOutput`.
- **Data-parallel execution** — one engine process per GPU slice across the task's GPUs.
- **Sampling / structured output** — import-free configuration for constrained decoding.

## Installation

```bash
pip install --pre flyteplugins-deepinfer
```

DeepInfer itself is **not on PyPI** — it must be built from source (Python 3.12,
CUDA 13 / CANN 9.2, GCC >= 13). Most teams bake a prebuilt DeepInfer into a base
image and layer this plugin on:

```python
import flyte

image = (
    flyte.Image.from_base("your-registry/deepinfer:sm100")
    .clone(name="deepinfer", extendable=True)  # required before adding layers
    .with_pip_packages("flyteplugins-deepinfer", pre=True)
)
```

See `deepinfer_image_from_source` in `flyteplugins.deepinfer` if you want the plugin
to compile DeepInfer from source into the image (heavy).

## Examples

| Example | When to use it |
| --- | --- |
| [`deepinfer_offline_inference.py`](examples/deepinfer_offline_inference.py) | Repeated small batches on **reusable** warm containers; the engine is cached per container. |
| [`deepinfer_data_parallel.py`](examples/deepinfer_data_parallel.py) | One large batch on one node, **one engine per GPU** for maximum throughput. |
| [`deepinfer_structured_extraction.py`](examples/deepinfer_structured_extraction.py) | **Structured output**: constrain generation to a Pydantic model's JSON schema and return validated objects. |
| [`deepinfer_batch_pipeline.py`](examples/deepinfer_batch_pipeline.py) | End-to-end pipeline: **cache the checkpoint** in a `flyte.io.Dir`, read prompts from a JSONL `File`, **fan out shards** over independent GPU nodes, write results back to a `File`. |
| [`deepinfer_multinode.py`](examples/deepinfer_multinode.py) | One expert-parallel pool **spanning several nodes** with `ClusteredTaskEnvironment`, starting from a **prebuilt** image. |

## Engine layouts

DeepInfer sizes an engine with three numbers: `ep_size` (devices the MoE experts are
split across), `cp_size` (devices per engine), and `data_parallel_size = ep_size //
cp_size` (number of engines). Every engine runs in its own process pinned to its own
devices, so a task uses one of two layouts:

- **One engine over all GPUs** — `create_llm(model, cp_size=get_actual_num_gpus())`.
  The engine lives in the task process, so it can be cached on a reusable container.
- **One engine per GPU** — `run_engines(worker, dp_size=get_actual_num_gpus())`, with
  `create_llm(model, dp_rank)` inside `worker`. Highest throughput for a large batch.

`create_llm` defaults `ep_size` to every device in the job and `data_parallel_size`
to `ep_size // cp_size`, so you rarely set either. Engines are started with `spawn`, so
`run_engines` workers must be module-level functions; bind their inputs with
`functools.partial`.

`generate` returns, per prompt, a list of per-token results. Decode them with the
engine's tokenizer: `tokenizer.decode([t.token_id for t in output],
skip_special_tokens=True)`.

## Quickstart: offline inference in a task

From `examples/deepinfer_offline_inference.py`. DeepInfer is heavy to cold start
(large model + CUDA warm-up), so the environment is reusable and the engine is cached
with `functools.lru_cache`, so it is built once per warm container:

```python
import functools

import flyte
from flyteplugins.deepinfer import (
    create_llm, deepinfer_image_from_source, get_actual_num_gpus, sampling_params, setup_common_env,
)

env = flyte.TaskEnvironment(
    name="deepinfer-batch",
    # 2+ replicas avoid starvation; idle_ttl keeps a warm pool ready.
    reusable=flyte.ReusePolicy(replicas=(2, 4), idle_ttl=300),
    resources=flyte.Resources(gpu=8),
    image=deepinfer_image_from_source(),
)

@functools.lru_cache(maxsize=1)
def _build_llm(model: str):
    # Env is configured, model resolved, GPUs pinned, deepinfer imported, in that order.
    setup_common_env()
    return create_llm(model, cp_size=get_actual_num_gpus())  # one engine over all GPUs

@env.task
async def score(prompts: list[str], model: str = "deepseek-ai/DeepSeek-V4.1-Flash-1017") -> list[str]:
    from deepinfer.sampling_params import SamplingParams

    llm = _build_llm(model)  # cached on warm containers
    params = SamplingParams(**sampling_params(temperature=0, max_tokens=256))
    outputs = llm.generate(prompts=prompts, sampling_params=[params] * len(prompts), use_tqdm=False)
    tokenizer = llm.get_tokenizer()
    return [tokenizer.decode([t.token_id for t in o], skip_special_tokens=True) for o in outputs]

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.run(score, prompts=["Hello!", "Tell me a joke."])
```

## Reusable containers

`reusable=flyte.ReusePolicy(...)` keeps a pool of warm containers alive across task
invocations, which is a good fit for inference: the DeepInfer engine (model weights +
CUDA kernels + async scheduler) is far more expensive to build than a single
`generate` call, so you want it to survive between runs. Because a warm container
reuses the same Python process, cache the engine with `functools.lru_cache` (as above)
so `create_llm` runs only on a cold container. Later invocations hit the cache
and skip the model load entirely. Only an engine built in the task process can be
cached this way; `run_engines` engines live in child processes that exit with the call.

Reusable applies to **single-node** `TaskEnvironment`s. It is **not available on
`ClusteredTaskEnvironment`** (multi-node): a clustered task is an ephemeral JobSet
that is torn down after each run, and the SDK rejects `reusable` with `ValueError:
ClusteredTaskEnvironment does not support reusable environments`. For multi-node
DeepInfer the warm-up cost has to be paid per run.

## Data-parallel batch inference

From `examples/deepinfer_data_parallel.py`: one engine per GPU, each scoring a
strided shard of the prompts, with results collected back in input order:

```python
import functools

import flyte
from flyteplugins.deepinfer import create_llm, get_actual_num_gpus, run_engines, setup_common_env

def generate_shard(model: str, prompts: list[str], dp_size: int, dp_rank: int, out) -> None:
    from deepinfer.sampling_params import SamplingParams

    llm = create_llm(model, dp_rank)
    indices = list(range(dp_rank, len(prompts), dp_size))
    outputs = llm.generate(
        prompts=[prompts[i] for i in indices],
        sampling_params=[SamplingParams(max_tokens=512) for _ in indices],
    )
    tokenizer = llm.get_tokenizer()
    out.extend(
        (i, tokenizer.decode([t.token_id for t in o], skip_special_tokens=True))
        for i, o in zip(indices, outputs)
    )

@env.task
async def batch_generate(prompts: list[str], model: str) -> list[str]:
    results = run_engines(
        functools.partial(generate_shard, model, prompts),
        dp_size=get_actual_num_gpus(),
        initializer=setup_common_env,
    )
    texts = dict(results)
    return [texts[i] for i in range(len(prompts))]
```

To scale past one node while keeping the model on a single node, run several of these
tasks concurrently over shards of the input, as in `examples/deepinfer_batch_pipeline.py`.

## Multi-node: `ClusteredTaskEnvironment`

DeepInfer is multi-node by design: its MoE expert-parallel pool can span **all** the
GPUs across nodes, and each node runs its own per-node launcher (which spawns the
local engine processes) using a scheduler-injected topology. Flyte's
`ClusteredTaskEnvironment` (a Kubernetes JobSet) is the right vessel:
`clustered_deepinfer_environment` wires it to DeepInfer's one-node-launcher-per-pod
model. From `examples/deepinfer_multinode.py`:

```python
cluster = clustered_deepinfer_environment(
    nodes=2,  # job spans 2 pods / DeepInfer nodes
    image=flyte.Image.from_base("ghcr.io/your-org/deepinfer:sm100")
    .clone(name="deepinfer", extendable=True)
    .with_pip_packages("flyteplugins-deepinfer", pre=True),
    resources=flyte.Resources(gpu=8),  # per node
)
driver = flyte.TaskEnvironment(name="driver", image=..., depends_on=[cluster])

@cluster.task
async def generate_on_cluster(prompts: list[str], model: str, out_dir: str) -> None:
    # Runs once per node; run_engines spawns only this node's DP ranks.
    dp_size = int(os.environ["WORLD_SIZE"]) * get_actual_num_gpus()
    results = run_engines(functools.partial(generate_shard, model, prompts), dp_size=dp_size,
                          initializer=setup_common_env)
    shard = flyte.io.File.from_existing_remote(f"{out_dir}/node-{os.environ['RANK']}.json")
    async with shard.open("wb") as fh:
        await fh.write(json.dumps(results).encode())

@driver.task
async def multinode_generate(prompts: list[str], model: str) -> list[str]:
    out_dir = flyte.io.Dir.new_remote("deepinfer-results")
    await generate_on_cluster(prompts, model, out_dir.path)
    ...  # read every node-*.json under out_dir and reassemble by index
```

How the pieces line up:

- `replicas = nodes` and `nproc_per_node = 1`: one *node-launcher* process per pod,
  owning all of that node's GPUs. That is DeepInfer's contract: it spawns its own
  engines and sets up its own cross-node rendezvous.
- The helper defaults to the `JaxRun` clustered runtime. Despite the JAX name it is
  the precise fit here: one process per pod, no launcher binary, and it exports
  `RANK`/`WORLD_SIZE`/`MASTER_ADDR`/`MASTER_PORT` (the same values torchrun would set)
  without running an unused torchrun rendezvous. `TorchRun()` with `nproc_per_node=1`
  is a drop-in alternative; no `jax_initialize` is needed either way for DeepInfer.
- `run_engines` / `launch_dp` already split the global DP ranks across nodes using
  `RANK`/`WORLD_SIZE`, and `create_llm` defaults `ep_size = WORLD_SIZE × GPUs-per-node`
  so EP spans the whole job.
- **Only rank 0's return value becomes the task output**, and `run_engines` returns
  just the local node's ranks, so each node writes its results to shared storage and
  a regular driver task merges them. The driver is also required because a clustered
  task cannot launch other tasks.

## Structured output

```python
from flyteplugins.deepinfer import sampling_params, structured_output

params = sampling_params(
    temperature=0,
    max_tokens=128,
    so_args=structured_output("json", '{"type":"object","properties":{...}}'),
)
```

`structured_output` accepts `regex`, `json`, and `ebnf`, plus optional
`trigger_token_ids` / `bound_token_id` controls. Expand the dict into DeepInfer's
`deepinfer.sampling_params.SamplingParams(**params)`. `sampling_params` only includes the
fields you set; everything else falls back to DeepInfer's own `SamplingParams`
defaults (note upstream's default `max_tokens` is small).

## Configuration and defaults

Nothing in the plugin pins a model, a node shape, or a toolchain version you cannot
change. The defaults assume the release-supported setup (DeepSeek-V4.1-Flash on an
8-GPU SM100 node) and are all overridable:

- `create_llm(model, ...)` — the model is always yours to pass (HF id, local path, or
  mounted `flyte.io.Dir`). `ep_size` defaults to every device in the job.
- `setup_common_env(...)` — each DeepInfer runtime flag is a keyword argument
  (defaults from DeepInfer's own example config); env vars you set yourself win.
- `clustered_deepinfer_environment(..., resources=...)` — per-node resources are
  whatever you request; the helper adds none of its own.
- `default_deepinfer_image(python_version=, cuda_version=, torch_version=)` and
  `deepinfer_image_from_source(repo=, ref=, install_dir=, ...)` — toolchain versions
  and the DeepInfer source default to what upstream's build expects. Pin `ref` to a
  tag for reproducible images.

## API

| Symbol | Purpose |
| --- | --- |
| `create_llm(model, engine_rank=0, *, ep_size=None, cp_size=1, env=None, **llm_kwargs)` | Build a DeepInfer `LLM` with the correct env-before-import ordering. |
| `resolve_model_path(model)` | HF id / local checkpoint / mounted flyte dir → local config dir. |
| `setup_common_env(...)` | Defaults for the DeepInfer runtime flags (nextn, EPLB, sampler). |
| `setup_cuda_visible_devices(rank, cp_size, device_count)` | Pin the current engine to its GPU slice. |
| `launch_dp(entry, dp_size=None)` / `run_dp(fn, dp_size)` | Multi-process DP launchers (upstream-compatible). |
| `run_engines(fn, dp_size, initializer=None)` | Run one engine per local DP rank (`spawn`) and collect results in rank order. |
| `sampling_params(**kw)` / `structured_output(type, schema, **extra)` | Import-free sampling / constrained decoding config. |
| `clustered_deepinfer_environment(nodes, image, resources=None, runtime=None, ...)` | Multi-node `ClusteredTaskEnvironment` (JobSet) for DeepInfer. |
| `default_deepinfer_image(...)` / `deepinfer_image_from_source(...)` | Container image helpers (versions, repo, and ref are parameters). |

## Limitations

- **Hardware-specific.** DeepInfer runs only on NVIDIA SM100 GPUs (CUDA 13) and
  Huawei Ascend 950 NPUs (CANN 9.2); the engine cannot start without them.
- **Not a serving plugin.** This release of DeepInfer is offline/batch only — there is
  no OpenAI-compatible HTTP server. For serving, pair it with an `AppEnvironment`
  from a serving-focused plugin once an upstream server exists.
- **Built from source.** No DeepInfer PyPI wheel exists; the image must contain a
  built `deepinfer` package.
