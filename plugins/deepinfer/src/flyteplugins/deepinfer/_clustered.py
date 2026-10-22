"""Multi-node DeepInfer on a Flyte `ClusteredTaskEnvironment`.

DeepInfer is multi-node by design: its MoE expert-parallel (EP) pool and context
parallel (CP) span **all** the GPUs across nodes (`ep_size = WORLD_SIZE x
GPUs-per-node`), and it launches with its own per-node launcher rather than one
process per GPU. On each node the task body runs exactly one *node-launcher*
process, which:

1. reads the scheduler-injected topology (`RANK` = node index, `WORLD_SIZE` =
   node count, `MASTER_ADDR`/`MASTER_PORT` = the cross-node rendezvous), and
2. spawns one engine process per local DP engine (`flyteplugins.deepinfer.run_dp` /
   `run_engines`), each pinned to its own GPU slice.

Flyte's `ClusteredTaskEnvironment` emits a Kubernetes JobSet for exactly this kind
of distributed job. Its **JAX runtime** (`JaxRun`) is the precise fit even though
we are not running JAX: it runs one process per pod (`nproc_per_node=1`, enforced),
gives that process every local accelerator, and exports `RANK`/`WORLD_SIZE`/
`MASTER_ADDR`/`MASTER_PORT` without a launcher binary or an unused torchrun
rendezvous — then execs the task body directly. `Clustered_deepinfer_environment`
wires that up; the task body calls `flyteplugins.deepinfer.run_dp`/`run_engines`:

```python
import flyte
from flyteplugins.deepinfer import clustered_deepinfer_environment, run_engines, create_llm, setup_common_env

env = clustered_deepinfer_environment(nodes=4, image=DEFAULT_DEEPINFER_IMAGE, gpu_per_node=8)

@env.task(timeout=flyte.Timeout(max_runtime=7200))
async def score_all(batches: list[list[str]]) -> list[str]:
    # One process per pod; launch_dp/run_engines split the DP ranks across nodes.
    def per_engine(dp_size: int, dp_rank: int, out) -> None:
        llm = create_llm("deepseek-ai/DeepSeek-V4.1-Flash-1017", dp_rank, ep_size=dp_size)
        outs = llm.generate(batches[dp_rank], ..., use_tqdm=False)
        out.extend(o[0].text for o in outs)

    return run_engines(per_engine, dp_size=len(batches), initializer=setup_common_env)
```

`TorchRun()` with `nproc_per_node=1` is a drop-in alternative (torchrun is a
DeepInfer dependency anyway), but it spins up an unused torchrun rendezvous and
requires the torchrun binary; `JaxRun()` is lighter and matches DeepInfer's
one-process-per-node model exactly. Neither needs `jax_initialize` for a
non-JAX engine like DeepInfer.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from flyte import Image, Resources
from flyte.clustered import ClusteredTaskEnvironment, JaxRun, Runtime


def clustered_deepinfer_environment(
    *,
    nodes: int,
    image: Union[str, Image],
    name: str = "deepinfer",
    gpu_per_node: Optional[Union[int, str]] = None,
    cpu: str = "48",
    memory: str = "512Gi",
    runtime: Runtime = JaxRun(),
    **env_kwargs: Any,
) -> ClusteredTaskEnvironment:
    """A `ClusteredTaskEnvironment` shaped to DeepInfer's one-node-launcher-per-pod model.

    Each pod is one DeepInfer *node*: one process owns all of that pod's GPUs and runs
    DeepInfer's per-node launcher (`run_dp`/`run_engines`), which spawns the local
    engine processes. `replicas == nodes` and `nproc_per_node == 1` are therefore
    fixed by contract; override `nproc_per_node` only if you deliberately run more than
    one node-launcher per pod.

    Args:
        nodes: Number of DeepInfer nodes == number of pods (JobSet replicas).
        image: Container image (expected to contain a built `deepinfer` package).
        name: Task/environment name.
        gpu_per_node: GPUs per node/pod. Pass an int, or an accelerator token such as
            `"H200:8"`. Left unset when None (no resource request is emitted).
        cpu: CPU request per node.
        memory: Memory request per node.
        runtime: `flyte.clustered.JaxRun` (default) or `TorchRun` — the per-node
            launcher. See the module docstring for why JaxRun fits DeepInfer despite the
            name.
        **env_kwargs: Extra `ClusteredTaskEnvironment` fields (`env_vars`, `secrets`,
            `failure_policy`, `timeout` via the decorator, etc.).

    Returns:
        A `ClusteredTaskEnvironment` ready to decorate a task whose body calls
        `run_dp` / `run_engines`.
    """
    if nodes < 1:
        raise ValueError("nodes must be >= 1")

    resources: Optional[Resources] = None
    if gpu_per_node is not None:
        resources = Resources(cpu=cpu, memory=memory, gpu=gpu_per_node)  # type: ignore[arg-type]

    env_kwargs.setdefault("nproc_per_node", 1)
    if env_kwargs.get("nproc_per_node", 1) != 1:
        # DeepInfer wants one node-launcher per pod; more than one would double-spawn
        # engines over the same GPUs. Guard loudly so a dropped value is not silent.
        raise ValueError(
            "DeepInfer runs one node-launcher process per pod; nproc_per_node must be 1 "
            "(set it on the environment's resources/launcher instead)."
        )

    return ClusteredTaskEnvironment(
        name=name,
        replicas=nodes,
        image=image,
        resources=resources,
        runtime=runtime,
        **env_kwargs,
    )


__all__ = ["clustered_deepinfer_environment"]
