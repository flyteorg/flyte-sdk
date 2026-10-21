"""Run DeepInfer engines across a task's GPUs.

DeepInfer's model is *multi-process data parallelism*: each data-parallel rank runs
its own engine process (with its own `CUDA_VISIBLE_DEVICES` slice), so that an
8-GPU task can serve one big MoE pool split as 8 prefill engines, or 2 engines each
over 4 context-parallel GPUs. The helpers here mirror the launch machinery in
DeepInfer's own examples, plus a `run_engines` that collects each engine's return
value back into the task process so a Flyte task can emit one result list.

```python
import flyte
from flyteplugins.deepinfer import create_llm, launch_dp, setup_common_env

@env.task(resources=flyte.Resources(cpu="48", memory="512Gi", gpu="SM100:8"))
async def batch_score(prompts_per_engine: list[list[str]]) -> list[list[str]]:
    setup_common_env()
    results: list[list[str]] = []

    def per_engine(dp_size: int, dp_rank: int) -> None:  # runs in a child process
        llm = create_llm("deepseek-ai/DeepSeek-V4.1-Flash-1017", dp_rank, ep_size=dp_size)
        outs = llm.generate(prompts_per_engine[dp_rank], ..., use_tqdm=False)
        results[dp_rank] = [o[0].text for o in outs]

    launch_dp(per_engine)          # spawns one child per local DP rank
    return results
```

`launch_dp`/`run_dp` are faithful ports of DeepInfer's own `_launch.py` (they fork
nothing into the Flyte task's result — use `run_engines` when you want return values).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from multiprocessing import Queue, get_context

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
    child process and appends its result(s) to `out` (a `multiprocessing.Queue`); each
    queue's items are drained in order and concatenated. With `dp_size == 1` the call
    runs in-process and `out` is a plain list; the worker contract is identical either
    way, so task code does not branch on deployments.

    Args:
        main_fn: Called as `main_fn(dp_size, dp_rank, out)` in each child.
        dp_size: Total data-parallel engines.
        initializer: Optional callable run in each child before `main_fn` (e.g.
            `flyteplugins.deepinfer.setup_common_env`).

    Returns:
        A list of every item each engine put on its queue, in rank order.
    """
    if dp_size == 1:
        out: list = []
        initializer() if initializer else None
        main_fn(dp_size, 0, out)
        return out

    ctx = get_context("spawn")
    queues: list[Queue] = [ctx.Queue() for _ in range(dp_size)]
    node_rank = int(os.environ.get(const.RANK, "0"))
    num_nodes = int(os.environ.get(const.WORLD_SIZE, "1"))
    assert dp_size % num_nodes == 0, "dp_size must be divisible by WORLD_SIZE"

    procs = []
    for dp_rank in range(dp_size):
        if dp_rank // (dp_size // num_nodes) != node_rank:
            continue
        target = main_fn if initializer is None else _init_then(initializer, main_fn)
        procs.append(ctx.Process(target=target, args=(dp_size, dp_rank, queues[dp_rank]), daemon=False))
    for p in procs:
        p.start()

    # Drain each engine's queue while its processes are still alive, so a result larger
    # than the queue buffer never deadlocks (a blocked `put` in the child needs the
    # parent to keep pulling). Once every local process has exited we pull whatever is
    # left, then fail loudly if any rank crashed.
    results: list = []
    while True:
        still_alive = [p for p in procs if p.is_alive()]
        got = False
        for q in queues:
            try:
                results.append(q.get(timeout=0.1))
                got = True
            except Exception:
                pass
        if not got and not still_alive:
            break

    for p in procs:
        p.join()
    failed = [i for i, p in enumerate(procs) if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"DeepInfer engines failed: {failed}")
    return results


def _init_then(initializer: Callable[[], None], main_fn: Callable[[int, int, list], None]):
    import functools

    @functools.wraps(main_fn)
    def wrapped(dp_size: int, dp_rank: int, out) -> None:
        initializer()
        main_fn(dp_size, dp_rank, out)

    return wrapped
