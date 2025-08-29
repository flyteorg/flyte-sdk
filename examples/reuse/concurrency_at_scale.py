import asyncio
import os
import pathlib
from pathlib import Path
import time

import flyte
import flyte.errors
from flyte._image import PythonWheels


actor_dist_folder = Path("/Users/ytong/go/src/github.com/unionai/flyte/fasttask/worker-v2/dist")
wheel_layer = PythonWheels(wheel_dir=actor_dist_folder, package_name="unionai-reuse")
base = flyte.Image.from_debian_base()
actor_image = base.clone(addl_layer=wheel_layer)
# actor_image = flyte.Image.from_debian_base().with_pip_packages("unionai-reuse==0.1.4", pre=True)

env = flyte.TaskEnvironment(
    name="oomer_parent_actor",
    # resources=flyte.Resources(cpu=1, memory="250Mi"),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=actor_image,
    reusable=flyte.ReusePolicy(
        replicas=1,
        idle_ttl=60,
        concurrency=100,
    ),
)


@env.task
async def concurrent_leaf(x: int):
    print(f"Leaf task got {x=}", flush=True)
    print(f"Leaf task {x=} finishing", flush=True)


@env.task
async def always_succeeds() -> int:
    # await asyncio.sleep(60)
    return 42


@env.task
async def concurrency_parent() -> int:
    print("Stating concurrency_parent main parent task", flush=True)
    # await asyncio.sleep(10)
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

    last_task_start = time.time()
    res = await always_succeeds()
    last_task_end = time.time()
    print(f"A0 finished!!! {last_task_start} --> {last_task_end}", flush=True)
    return res


if __name__ == "__main__":
    flyte.init_from_config("/Users/ytong/.flyte/config-k3d.yaml")

    run = flyte.run(concurrency_parent)
    print(run.url)
    run.wait(run)
