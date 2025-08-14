import asyncio
import datetime
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


actor_env = flyte.TaskEnvironment(
    name="concurrent_mapper",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=actor_image,
    reusable=flyte.ReusePolicy(
        replicas=1,
        idle_ttl=60,
        concurrency=200,
    ),
)

parent_env = flyte.TaskEnvironment(
    name="map_parent",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
    depends_on=[actor_env],
)


@actor_env.task
async def concurrent_leaf(x: int) -> tuple[int, datetime.datetime]:
    print(f"Leaf task got {x=}", flush=True)
    return x, datetime.datetime.now()


@parent_env.task
async def map_parent() -> tuple[list[int], list[datetime.datetime]]:
    print("Starting concurrency_parent main parent task...", flush=True)
    tasks = [concurrent_leaf(x=i) for i in range(50)]
    print(f"Parent task will run {len(tasks)} concurrent tasks", flush=True)
    start_time = time.time()
    res = await asyncio.gather(*tasks)
    end_time = time.time()
    print(f"Map parent returning. Elapsed {start_time=} -> {end_time=} = {end_time - start_time} seconds", flush=True)
    results = ([r[0] for r in res], [r[1] for r in res])
    return results


if __name__ == "__main__":
    flyte.init_from_config("/Users/ytong/.flyte/config-k3d.yaml")
    run = flyte.run(map_parent)
    # print(run.url)
    # run.wait(run)
