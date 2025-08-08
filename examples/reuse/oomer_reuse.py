import asyncio
import os
import pathlib
from pathlib import Path

import flyte
import flyte.errors
from flyte._image import PythonWheels

PATH_TO_FASTTASK_WORKER = pathlib.Path("/Users/ytong/go/src/github.com/unionai/flyte/fasttask/worker-v2")

actor_dist_folder = Path("/Users/ytong/go/src/github.com/unionai/flyte/fasttask/worker-v2/dist")
wheel_layer = PythonWheels(wheel_dir=actor_dist_folder, package_name="unionai-reuse")
base = flyte.Image.from_debian_base()
actor_image = base.clone(addl_layer=wheel_layer)

# hopefully this makes it not need to be rebuilt every time
# object.__setattr__(actor_image, "_tag", "9043815457d6422e4adb4fb83c5d3c5a")
# ghcr.io/flyteorg/flyte:9043815457d6422e4adb4fb83c5d3c5a

env = flyte.TaskEnvironment(
    name="oomer_parent_actor",
    # resources=flyte.Resources(cpu=1, memory="250Mi"),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=actor_image,
    reusable=flyte.ReusePolicy(
        replicas=1,
        idle_ttl=60,
        concurrency=8,
    ),
)

leaf_env = flyte.TaskEnvironment(
    name="leaf_env",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
)


@env.task
async def oomer(x: int):
    print("Leaf (oomer) Environment Variables:", os.environ, flush=True)
    print("About to allocate a large list... should oom", flush=True)
    await asyncio.sleep(1)
    large_list = [0] * 100000000
    print(len(large_list))


@env.task
async def no_oom(x: int):
    print("Leaf (non-oom) Environment Variables:", os.environ, flush=True)
    print(f"Non-ooming task got {x=}", flush=True)
    await asyncio.sleep(1)
    print(f"Non-ooming task {x=} finishing", flush=True)


@env.task
async def always_succeeds() -> int:
    await asyncio.sleep(1)
    return 42


@env.task
async def failure_recovery() -> int:
    print("A0 (failure recovery) Environment Variables:", os.environ, flush=True)
    try:
        # await oomer(2)
        tasks = []
        for i in range(4):
            tasks.append(no_oom(x=i))
        import time

        start_time = time.time()
        print(
            f"About to gather {len(tasks)} tasks at time index"
            f" {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}",
            flush=True,
        )
        results = await asyncio.gather(*tasks)
        print(f"All tasks completed successfully: {results}", flush=True)
    except flyte.errors.OOMError as e:
        print(f"Failed with oom trying with more resources: {e}, of type {type(e)}, {e.code}")
        try:
            await oomer.override(resources=flyte.Resources(cpu=1, memory="1Gi"))(5)
        except flyte.errors.OOMError as e:
            print(f"Failed with OOM Again giving up: {e}, of type {type(e)}, {e.code}")
            raise e
    finally:
        # await always_succeeds()
        print("In finally...", flush=True)

    res = await always_succeeds()
    print("A0 finished!!!!!!!!!!!!!", flush=True)
    return res


if __name__ == "__main__":
    flyte.init_from_config("/Users/ytong/.flyte/config-k3d.yaml")

    run = flyte.run(failure_recovery)
    print(run.url)
    run.wait(run)
