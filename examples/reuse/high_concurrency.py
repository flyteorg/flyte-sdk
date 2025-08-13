import asyncio
import os
import pathlib
from pathlib import Path
import time

import flyte
import flyte.errors
from flyte._image import PythonWheels

PATH_TO_FASTTASK_WORKER = pathlib.Path("/Users/ytong/go/src/github.com/unionai/flyte/fasttask/worker-v2")

actor_dist_folder = Path("/Users/ytong/go/src/github.com/unionai/flyte/fasttask/worker-v2/dist")
wheel_layer = PythonWheels(wheel_dir=actor_dist_folder, package_name="unionai-reuse")
base = flyte.Image.from_debian_base()
actor_image = base.clone(addl_layer=wheel_layer)


env = flyte.TaskEnvironment(
    name="oomer_parent_actor",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
    # resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=actor_image,
    reusable=flyte.ReusePolicy(
        replicas=1,
        idle_ttl=60,
        concurrency=200,
    ),
)

leaf_env = flyte.TaskEnvironment(
    name="leaf_env",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
)


@env.task
async def concurrent_leaf(x: int):
    print(f"Leaf task got {x=}", flush=True)
    # await asyncio.sleep(1)
    print(f"Leaf task {x=} finishing", flush=True)


@env.task
async def always_succeeds() -> int:
    # await asyncio.sleep(1)
    return 42


@env.task
async def concurrency_parent() -> int:
    print("Stating concurrency_parent main parent task", flush=True)
    try:
        tasks = []
        for i in range(50):
            tasks.append(concurrent_leaf(x=i))

        start_time = time.time()
        print(
            f"About to gather {len(tasks)} tasks at time index"
            f" {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}",
            flush=True,
        )
        results = await asyncio.gather(*tasks)
        print(f"All tasks completed successfully.", flush=True)
    finally:
        print("In finally...", flush=True)

    res = await always_succeeds()
    print("A0 finished!!!", flush=True)
    return res


if __name__ == "__main__":
    flyte.init_from_config("/Users/ytong/.flyte/config-k3d.yaml")

    run = flyte.run(concurrency_parent)
    print(run.url)
    run.wait(run)
