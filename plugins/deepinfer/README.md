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
from flyteplugins.deepinfer import default_deepinfer_image

image = flyte.Image.from_base("your-registry/deepinfer:sm100").with_pip_packages(
    "flyteplugins-deepinfer", pre=True
)
```

See `deepinfer_image_from_source` in `flyteplugins.deepinfer` if you want the plugin
to compile DeepInfer from source into the image (heavy).

## Quickstart — offline inference in a task

`examples/deepinfer_offline_inference.py`:

# DeepInfer is heavy to cold start (large model + CUDA warm-up). Mark the environment
# reusable so a pool of warm containers stays alive across invocations, and cache the
# engine with functools.lru_cache so it is built once per warm container.
import functools

import flyte
from flyteplugins.deepinfer import create_llm, sampling_params, default_deepinfer_image

env = flyte.TaskEnvironment(
    name="deepinfer-batch",
    # 2+ replicas avoid starvation; idle_ttl keeps a warm pool ready.
    reusable=flyte.ReusePolicy(replicas=(2, 4), idle_ttl=300),
    resources=flyte.Resources(cpu="48", memory="512Gi", gpu=8),
    image=default_deepinfer_image(),
)

@functools.lru_cache(maxsize=1)
def _build_llm(model: str):
    # Env is configured, model resolved, GPUs pinned, deepinfer imported — in that order.
    return create_llm(model, engine_rank=0, ep_size=8)

@env.task
async def score(prompts: list[str]) -> list[str]:
    llm = _build_llm("deepseek-ai/DeepSeek-V4.1-Flash-1017")  # cached on warm containers
    params = sampling_params(temperature=0, max_tokens=256)
    outputs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
    return [o[0].text for o in outputs]

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.serve(score)
```

## Reusable containers

`reusable=flyte.ReusePolicy(...)` keeps a pool of warm containers alive across task
invocations, which is a good fit for inference: the DeepInfer engine (model weights +
CUDA kernels + async scheduler) is far more expensive to build than a single
`generate` call, so you want it to survive between runs. Because a warm container
reuses the same Python process, cache the engine with `functools.lru_cache` (as above)
so `create_llm` runs only on a cold container — subsequent invocations hit the cache
and skip the model load entirely.

Reusable applies to **single-node** `TaskEnvironment`s (quickstart, data-parallel
above). It is **not available on `ClusteredTaskEnvironment`** (multi-node): a clustered
task is an ephemeral JobSet that is torn down after each run, and the SDK rejects
`reusable` with `ValueError: ClusteredTaskEnvironment does not support reusable
environments`. For multi-node DeepInfer the warm-up cost has to be paid per run.

## Data-parallel batch inference

DeepInfer runs one engine per data-parallel rank. Use `launch_dp` (ports DeepInfer's
own launcher) or `run_engines` when you also want each engine's results returned:

```python
import flyte
from flyteplugins.deepinfer import create_llm, run_engines, setup_common_env

@env.task(resources=flyte.Resources(cpu="48", memory="512Gi", gpu=8))
async def score_all(batches: list[list[str]]) -> list[str]:
    setup_common_env()

    def per_engine(dp_size: int, dp_rank: int, out) -> None:
        llm = create_llm("deepseek-ai/DeepSeek-V4.1-Flash-1017", dp_rank, ep_size=dp_size)
        outs = llm.generate(batches[dp_rank], sampling_params=..., use_tqdm=False)
        out.extend(o[0].text for o in outs)

    return run_engines(per_engine, dp_size=len(batches))
```

## Multi-node: `ClusteredTaskEnvironment`

DeepInfer is multi-node by design — its MoE expert-parallel pool spans **all** the
GPUs across nodes, and each node runs its own per-node launcher (which spawns the
local engine processes) using a scheduler-injected topology. Flyte's
`ClusteredTaskEnvironment` (a Kubernetes JobSet) is the right vessel:
`clustered_deepinfer_environment` wires it to DeepInfer's one-node-launcher-per-pod
model.

```python
import flyte
from flyteplugins.deepinfer import (
    clustered_deepinfer_environment, run_engines, create_llm, setup_common_env,
)

env = clustered_deepinfer_environment(
    nodes=4,  # job spans 4 pods / DeepInfer nodes
    image=default_deepinfer_image(),
    gpu_per_node=8,  # SM100 / Ascend GPUs per node
)

@env.task(timeout=flyte.Timeout(max_runtime=7200))
async def score_all(batches: list[list[str]]) -> list[str]:
    # Runs once per pod (once per node); run_engines spawns one engine per local DP
    # rank, using the injected RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT.
    def per_engine(dp_size: int, dp_rank: int, out) -> None:
        llm = create_llm("deepseek-ai/DeepSeek-V4.1-Flash-1017", dp_rank, ep_size=dp_size)
        outs = llm.generate(batches[dp_rank], sampling_params=..., use_tqdm=False)
        out.extend(o[0].text for o in outs)

    return run_engines(per_engine, dp_size=len(batches), initializer=setup_common_env)
```

How the pieces line up:

- `replicas = nodes` and `nproc_per_node = 1`: one *node-launcher* process per pod,
  owning all of that node's GPUs — exactly DeepInfer's contract (it spawns its own
  engines and sets up its own cross-node rendezvous).
- The helper defaults to the `JaxRun` clustered runtime. Despite the JAX name it is
  the precise fit here: one process per pod, no launcher binary, and it exports
  `RANK`/`WORLD_SIZE`/`MASTER_ADDR`/`MASTER_PORT` (the same values torchrun would set)
  without running an unused torchrun rendezvous. `TorchRun()` with `nproc_per_node=1`
  is a drop-in alternative — no `jax_initialize` is needed either way for DeepInfer.
- `run_engines` / `launch_dp` already split the global DP ranks across nodes using
  `RANK`/`WORLD_SIZE`, and `create_llm` sets `ep_size = WORLD_SIZE × GPUs-per-node`
  so EP spans the whole job.

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
`trigger_token_ids` / `bound_token_id` controls.

## API

| Symbol | Purpose |
| --- | --- |
| `create_llm(model, engine_rank, *, ep_size, cp_size=1, env=None, **llm_kwargs)` | Build a DeepInfer `LLM` with the correct env-before-import ordering. |
| `resolve_model_path(model)` | HF id / local checkpoint / mounted flyte dir → local config dir. |
| `setup_common_env(...)` | Defaults for the DeepInfer runtime flags (nextn, EPLB, sampler). |
| `setup_cuda_visible_devices(rank, cp_size, device_count)` | Pin the current engine to its GPU slice. |
| `launch_dp(entry, dp_size=None)` / `run_dp(fn, dp_size)` | Multi-process DP launchers (upstream-compatible). |
| `run_engines(fn, dp_size, initializer=None)` | Run one engine per DP rank and collect results. |
| `sampling_params(**kw)` / `structured_output(type, schema, **extra)` | Import-free sampling / constrained decoding config. |
| `clustered_deepinfer_environment(nodes, image, gpu_per_node, ...)` | Multi-node `ClusteredTaskEnvironment` (JobSet) for DeepInfer. |
| `default_deepinfer_image()` / `deepinfer_image_from_source()` | Container image helpers. |

## Limitations

- **Hardware-specific.** DeepInfer runs only on NVIDIA SM100 GPUs (CUDA 13) and
  Huawei Ascend 950 NPUs (CANN 9.2); the engine cannot start without them.
- **Not a serving plugin.** This release of DeepInfer is offline/batch only — there is
  no OpenAI-compatible HTTP server. For serving, pair it with an `AppEnvironment`
  from a serving-focused plugin once an upstream server exists.
- **Built from source.** No DeepInfer PyPI wheel exists; the image must contain a
  built `deepinfer` package.
