"""Multi-node DeepInfer on a Flyte `ClusteredTaskEnvironment`.

DeepInfer is multi-node by design: its MoE expert-parallel pool can span **all** the
GPUs across nodes (`ep_size = WORLD_SIZE x GPUs-per-node`), and it runs one launcher
per node rather than one process per GPU. On each node the task body runs exactly once;
`DeepInfer` reads the scheduler-injected topology (`RANK` = node index, `WORLD_SIZE` =
node count) and starts that node's engines.

Flyte's `ClusteredTaskEnvironment` emits a Kubernetes JobSet for exactly this kind of
distributed job. Its **JAX runtime** (`JaxRun`) is the precise fit even though we are
not running JAX: it runs one process per pod (`nproc_per_node=1`, enforced), gives that
process every local accelerator, and exports `RANK`/`WORLD_SIZE`/`MASTER_ADDR`/
`MASTER_PORT` without a launcher binary or an unused torchrun rendezvous.

```python
import flyte
from flyteplugins.deepinfer import DeepInfer, clustered_deepinfer_environment, deepinfer_image

cluster = clustered_deepinfer_environment(
    nodes=4, image=deepinfer_image(), resources=flyte.Resources(gpu="B200:8")
)
llm = DeepInfer("deepseek-ai/DeepSeek-V4.1-Flash-1017")  # one EP pool over all 32 GPUs

@cluster.task
async def generate_on_cluster(prompts: list[str], out_dir: str) -> None:
    results = await llm.generate.aio(prompts)  # this node's entries filled, others None
    ...  # write them under out_dir/node-{RANK}; see examples/deepinfer_multinode.py
```

Only rank 0's return value becomes the task output, and each node only produces its
own engines' results, so every node writes its results to shared storage and a regular
(non-clustered) driver task merges them. The driver is required anyway: only a regular
task can launch a clustered one.

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
    resources: Optional[Resources] = None,
    runtime: Optional[Runtime] = None,
    **env_kwargs: Any,
) -> ClusteredTaskEnvironment:
    """A `ClusteredTaskEnvironment` shaped to DeepInfer's one-node-launcher-per-pod model.

    Each pod is one DeepInfer *node*: one process owns all of that pod's GPUs, and
    `DeepInfer.generate` in the task body starts the local engine processes.
    `replicas == nodes` and `nproc_per_node == 1` are therefore fixed by contract;
    override `nproc_per_node` only if you deliberately run more than one launcher per pod.

    Args:
        nodes: Number of DeepInfer nodes == number of pods (JobSet replicas).
        image: Container image (expected to contain a built `deepinfer` package).
        name: Task/environment name.
        resources: Per-node (per-pod) resources, e.g. `flyte.Resources(gpu="B200:8")`.
            Every GPU on a node should be requested, since one process drives them all.
        runtime: `flyte.clustered.JaxRun()` (default) or `TorchRun()` — the per-node
            launcher. See the module docstring for why JaxRun fits DeepInfer despite the
            name.
        **env_kwargs: Extra `ClusteredTaskEnvironment` fields (`env_vars`, `secrets`,
            `failure_policy`, `timeout` via the decorator, etc.).

    Returns:
        A `ClusteredTaskEnvironment` ready to decorate a task whose body calls
        `DeepInfer.generate`.
    """
    if nodes < 1:
        raise ValueError("nodes must be >= 1")

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
        runtime=runtime if runtime is not None else JaxRun(),
        **env_kwargs,
    )


__all__ = ["clustered_deepinfer_environment"]
