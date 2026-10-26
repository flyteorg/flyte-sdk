"""Run DeepInfer engines across a task's GPUs.

DeepInfer's model is *multi-process data parallelism*: each data-parallel rank runs
its own engine process (with its own `CUDA_VISIBLE_DEVICES` slice), so that an
8-GPU task can serve one big MoE pool split as 8 prefill engines, or 2 engines each
over 4 context-parallel GPUs. The helpers here mirror the launch machinery in
DeepInfer's own examples, plus a `run_engines` that collects each engine's return
value back into the task process so a Flyte task can emit one result list.

```python
import functools

import flyte
from flyteplugins.deepinfer import create_llm, get_actual_num_gpus, run_engines, setup_common_env


def score_shard(model: str, prompts: list[str], dp_size: int, dp_rank: int, out) -> None:
    # Module-level so it pickles into the `spawn`ed child process.
    from deepinfer.sampling_params import SamplingParams

    llm = create_llm(model, dp_rank)  # ep_size / data_parallel_size default to the job
    shard = prompts[dp_rank::dp_size]
    outputs = llm.generate(prompts=shard, sampling_params=[SamplingParams(max_tokens=256)] * len(shard))
    tok = llm.get_tokenizer()
    out.extend(tok.decode([t.token_id for t in o], skip_special_tokens=True) for o in outputs)


@env.task(resources=flyte.Resources(gpu="SM100:8"))
async def batch_score(prompts: list[str], model: str) -> list[str]:
    return run_engines(
        functools.partial(score_shard, model, prompts),
        dp_size=get_actual_num_gpus(),  # one engine per GPU
        initializer=setup_common_env,
    )
```

`launch_dp`/`run_dp` are faithful ports of DeepInfer's own `_launch.py` (they return
nothing to the Flyte task — use `run_engines` when you want return values).
"""

from __future__ import annotations

import functools
import os
import queue
from collections.abc import Callable, Iterable
from multiprocessing import Queue, get_context
from typing import Any

from flyteplugins.deepinfer import _constants as const


def _node_dp_size() -> int:
    """Default DP size: one engine per visible device, across all nodes."""
    try:
        import torch

        return int(os.environ.get(const.WORLD_SIZE, "1")) * torch.cuda.device_count()
    except Exception:  # pragma: no cover - torch unavailable off-host
        from flyteplugins.deepinfer._bootstrapping import get_actual_num_gpus

        return int(os.environ.get(const.WORLD_SIZE, "1")) * get_actual_num_gpus()


def run_dp(main_fn: Callable[[int, int], None], dp_size: int) -> None:
    """Run `main_fn(dp_size, rank)` in a child process for each local DP rank.

    Distributes the global `[0, dp_size)` ranks across `WORLD_SIZE` nodes so only
    the ranks this node owns are spawned here (the same contract as DeepInfer's own
    `_launch.run_dp`). Child processes are started with the `spawn` context, which
    is required once the parent has imported CUDA/torch.

    Args:
        main_fn: Called with `(dp_size, dp_rank)` inside each child process.
        dp_size: Total number of data-parallel engines.
    """
    node_rank = int(os.environ.get(const.RANK, "0"))
    num_nodes = int(os.environ.get(const.WORLD_SIZE, "1"))
    assert dp_size % num_nodes == 0, "dp_size must be divisible by WORLD_SIZE"

    if dp_size == 1:
        main_fn(dp_size, 0)
        return

    ctx = get_context("spawn")
    procs = []
    for dp_rank in range(dp_size):
        if dp_rank // (dp_size // num_nodes) == node_rank:
            procs.append(ctx.Process(target=main_fn, args=(dp_size, dp_rank), daemon=False))
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    failed = [i for i, p in enumerate(procs) if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"DeepInfer engines failed: {failed}")


def launch_dp(entry: Callable[[int, int], None], dp_size: int | None = None) -> None:
    """Convenience around `run_dp` with single-node defaults.

    Defaults `RANK=0`/`WORLD_SIZE=1` and, when `dp_size` is None, one engine per
    visible GPU.

    Args:
        entry: Callable invoked as `entry(dp_size, dp_rank)` per engine.
        dp_size: Total DP engines; defaults to one per visible device across nodes.
    """
    os.environ.setdefault(const.RANK, "0")
    os.environ.setdefault(const.WORLD_SIZE, "1")
    if dp_size is None:
        dp_size = _node_dp_size()
    run_dp(entry, dp_size)


def run_engines(
    main_fn: Callable[[int, int, list], None],
    dp_size: int,
    *,
    initializer: Callable[[], None] | None = None,
) -> list:
    """Run one engine per DP rank and return each rank's emitted result.

    Unlike `run_dp`, the results each engine wants to surface are also brought back
    into the task process. The contract: `main_fn(dp_size, dp_rank, out)` runs in a
    child process and calls `out.append(item)` / `out.extend(items)`. With
    `dp_size == 1` the call runs in-process and `out` is a plain list; the worker
    contract is identical either way, so task code does not branch on deployments.

    Child processes use the `spawn` start method, so `main_fn` and `initializer` must
    be picklable: define them at module level and bind extra arguments with
    `functools.partial` rather than using closures or lambdas.

    On a multi-node job only this node's ranks run here, so the returned list holds
    this node's results only.

    Args:
        main_fn: Called as `main_fn(dp_size, dp_rank, out)` in each child.
        dp_size: Total data-parallel engines.
        initializer: Optional callable run in each child before `main_fn` (e.g.
            `flyteplugins.deepinfer.setup_common_env`).

    Returns:
        Every item this node's engines emitted, ordered by rank, then by emission order.
    """
    if dp_size == 1:
        out: list = []
        if initializer is not None:
            initializer()
        main_fn(dp_size, 0, out)
        return out

    node_rank = int(os.environ.get(const.RANK, "0"))
    num_nodes = int(os.environ.get(const.WORLD_SIZE, "1"))
    assert dp_size % num_nodes == 0, "dp_size must be divisible by WORLD_SIZE"
    local_ranks = [r for r in range(dp_size) if r // (dp_size // num_nodes) == node_rank]

    ctx = get_context("spawn")
    queues: dict[int, Queue] = {r: ctx.Queue() for r in local_ranks}
    target = main_fn if initializer is None else functools.partial(_init_then, initializer, main_fn)
    procs = [ctx.Process(target=target, args=(dp_size, r, _QueueSink(queues[r])), daemon=False) for r in local_ranks]
    for p in procs:
        p.start()

    # Drain each engine's queue while its processes are still alive, so a result larger
    # than the queue buffer never deadlocks (a blocked `put` in the child needs the
    # parent to keep pulling). Once every local process has exited we pull whatever is
    # left, then fail loudly if any rank crashed.
    per_rank: dict[int, list] = {r: [] for r in local_ranks}
    while True:
        still_alive = any(p.is_alive() for p in procs)
        got = False
        for r, q in queues.items():
            try:
                per_rank[r].extend(q.get(timeout=0.1))
                got = True
            except queue.Empty:
                pass
        if not got and not still_alive:
            break

    for p in procs:
        p.join()
    failed = [r for r, p in zip(local_ranks, procs) if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"DeepInfer engines failed: {failed}")
    return [item for r in local_ranks for item in per_rank[r]]


class _QueueSink:
    """List-like `out` handed to a child engine; forwards batches to the parent's queue."""

    def __init__(self, q: Queue) -> None:
        self._q = q

    def append(self, item: Any) -> None:
        self._q.put([item])

    def extend(self, items: Iterable[Any]) -> None:
        self._q.put(list(items))


def _init_then(
    initializer: Callable[[], None],
    main_fn: Callable[[int, int, list], None],
    dp_size: int,
    dp_rank: int,
    out: Any,
) -> None:
    # Module-level (not a closure) so `functools.partial` over it pickles under `spawn`.
    initializer()
    main_fn(dp_size, dp_rank, out)
