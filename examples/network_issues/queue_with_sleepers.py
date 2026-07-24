import asyncio
import random
from datetime import timedelta

import flyte
from flyte.extras import Sleep

sleep_env = flyte.TaskEnvironment(
    name="queue_drain_sleep_fanout_leaf",
    plugin_config=Sleep(),
    queue="drain-test-a1",
)

env = flyte.TaskEnvironment(
    name="queue-drain-sleep-parent",
    depends_on=[sleep_env],
)


@sleep_env.task
async def sleep_leaf(duration: timedelta) -> None:
    return None


@env.task
async def parent(
    n_children: int = 100,
    sleep_min: timedelta = timedelta(seconds=120),
) -> int:
    """
    Fan out n_children sleep_leaf leaves in parallel.
    """
    print(f"fanout_inputs {n_children=}, {sleep_min=}", flush=True)
    await asyncio.gather(
        *(sleep_leaf(duration=sleep_min + timedelta(seconds=random.randint(0, 50))) for _ in range(n_children))
    )
    print(f"Done. Total leaves: {n_children}", flush=True)
    return n_children


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(parent)
    print(run.name, run.url)
