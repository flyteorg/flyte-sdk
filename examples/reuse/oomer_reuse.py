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

# issue 2: OOMKilled pod doesn't get replaced with a new pod
#   should be fixed with autoscaling feature.

# issue 3: OOM doesn't get treated as a user error, because it gets caught as
#   task status not updated within grace period.
#   (if Task A and Task B are running on the same pod, and Task B ooms, do they both
#   get marked as user error? (we're not able to differentiate))

# issue 4: Logs are not differentiated between a) attempts that land on the same pod
#   b) different tasks running on the same pod (which wasn't a problem in v1).
#   someone added time boundaries for v1 to differentiate between attempts???

env = flyte.TaskEnvironment(
    name="oomer_parent_actor",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
    # resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=actor_image,
    reusable=flyte.ReusePolicy(
        replicas=2,
        idle_ttl=60,
        concurrency=2,
    ),
)

leaf_env = flyte.TaskEnvironment(
    name="leaf_env",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
)


@env.task
async def oomer(x: int):
    print("About to allocate a large list... should oom", flush=True)
    await asyncio.sleep(20)
    large_list = [0] * 100000000
    print(len(large_list))


@env.task
async def no_oom(x: int):
    print(f"Non-ooming task got {x=}", flush=True)
    # await asyncio.sleep(1)
    print(f"Non-ooming task {x=} finishing", flush=True)


@env.task
async def always_succeeds() -> int:
    # await asyncio.sleep(1)
    return 42


@env.task
async def failure_recovery() -> int:
    print("Stating oomer_reuse main parent task", flush=True)
    await asyncio.sleep(60)
    print("Slept 1 minute, trying oomer now", flush=True)
    try:
        await oomer(2)
        # tasks = []
        # for i in range(50):
        #     tasks.append(no_oom(x=i))

        # start_time = time.time()
        # print(
        #     f"About to gather {len(tasks)} tasks at time index"
        #     f" {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}",
        #     flush=True,
        # )
        # results = await asyncio.gather(*tasks)
        print(f"All tasks completed successfully.", flush=True)
    except flyte.errors.OOMError as e:
        print(f"Failed with oom trying with more resources: {e}, of type {type(e)}, {e.code}", flush=True)
        await asyncio.sleep(60)
        print("Waited 60 seconds, trying again with more resources...", flush=True)
        try:
            # issue 1: this is not creating a separate Actor environment.
            #   confirm the override shows up in the task_serde output for enqueueaction.
            await oomer.override(resources=flyte.Resources(cpu=1, memory="1Gi"))(5)
        except flyte.errors.OOMError as e:
            print(f"Failed with OOM Again giving up: {e}, of type {type(e)}, {e.code}")
            raise e
    finally:
        # await always_succeeds()
        print("In finally...", flush=True)

    res = await always_succeeds()
    print("A0 finished!!!", flush=True)
    return res


if __name__ == "__main__":
    flyte.init_from_config("/Users/ytong/.flyte/config-k3d.yaml")

    run = flyte.run(failure_recovery)
    print(run.url)
    run.wait(run)
