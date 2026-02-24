import asyncio
import random
import time

import flyte

env = flyte.TaskEnvironment("stress_fanout")


@env.task
async def leaf(path: list[int], sleep_sec: float, jitter_sec: float) -> dict:
    """
    Leaf task that prints its lineage, sleeps with jitter, and returns.
    """
    actual_sleep = max(0.0, sleep_sec + random.uniform(-jitter_sec, jitter_sec))
    depth = len(path)
    print(
        f"Leaf node at depth={depth} | path={path} | "
        f"sleeping {actual_sleep:.2f}s (base={sleep_sec}, jitter=±{jitter_sec})",
        flush=True,
    )
    start = time.monotonic()
    await asyncio.sleep(actual_sleep)
    elapsed = time.monotonic() - start
    return {"path": path, "sleep_requested": actual_sleep, "sleep_actual": elapsed}


@env.task
async def fanout(
    fanout_per_layer: list[int],
    layer: int,
    path: list[int],
    sleep_sec: float,
    jitter_sec: float,
) -> int:
    """
    Recursively fans out child tasks according to fanout_per_layer.

    At each layer, spawns fanout_per_layer[layer] children. When there are no
    more layers, delegates to a leaf task instead.
    """
    n_children = fanout_per_layer[layer]
    is_last_layer = layer + 1 >= len(fanout_per_layer)
    total_layers = len(fanout_per_layer)

    print(
        f"Fanout node at layer={layer}/{total_layers - 1} | path={path} | "
        f"spawning {n_children} {'leaves' if is_last_layer else 'children'}",
        flush=True,
    )

    if is_last_layer:
        # Spawn leaves
        coros = [leaf(path=path + [i], sleep_sec=sleep_sec, jitter_sec=jitter_sec) for i in range(n_children)]
        await asyncio.gather(*coros)
        print(
            f"Fanout node at layer={layer} | path={path} | "
            f"all {n_children} leaves completed",
            flush=True,
        )
        return n_children
    else:
        # Spawn intermediate fanout nodes
        coros = [
            fanout(
                fanout_per_layer=fanout_per_layer,
                layer=layer + 1,
                path=path + [i],
                sleep_sec=sleep_sec,
                jitter_sec=jitter_sec,
            )
            for i in range(n_children)
        ]
        results = await asyncio.gather(*coros)
        total = n_children + sum(results)
        print(
            f"Fanout node at layer={layer} | path={path} | "
            f"all {n_children} children completed | total descendant tasks: {total}",
            flush=True,
        )
        return total


@env.task
async def main(
    fanout_per_layer: list[int],
    sleep_sec: float = 1.0,
    jitter_sec: float = 0.5,
) -> int:
    """
    Stress test that creates a tree of tasks with configurable fan-out per layer.

    Args:
        fanout_per_layer: Number of children to spawn at each layer.
            e.g. [3000, 1, 100] means the root spawns 3000 children, each of
            those spawns 1 child, and each of those spawns 100 leaf tasks.
        sleep_sec: Base sleep duration for leaf tasks in seconds.
        jitter_sec: Maximum jitter (±) added to sleep duration.

    Returns:
        Total number of tasks spawned across all layers.
    """
    total = await fanout(
        fanout_per_layer=fanout_per_layer,
        layer=0,
        path=[],
        sleep_sec=sleep_sec,
        jitter_sec=jitter_sec,
    )
    print(f"Done. Total tasks spawned: {total}")
    return total


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext("local").run(main, fanout_per_layer=[1, 1, 2], sleep_sec=1.0, jitter_sec=0.5)
    print(run.outputs)
    # print(run.url)
