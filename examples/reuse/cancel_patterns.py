import asyncio
from pathlib import Path

import flyte
from flyte._image import PythonWheels

actor_dist_folder = Path("/Users/ytong/go/src/github.com/unionai/flyte/fasttask/worker-v2/dist")
wheel_layer = PythonWheels(wheel_dir=actor_dist_folder, package_name="unionai-reuse")
base = flyte.Image.from_debian_base()
actor_image = base.clone(addl_layer=wheel_layer)

actor_env = flyte.TaskEnvironment(
    name="reuse_cancel_patterns",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    reusable=flyte.ReusePolicy(
        replicas=1,
        idle_ttl=60,
        concurrency=5,
        scaledown_ttl=60,
    ),
    image=actor_image,
)

parent_env = flyte.TaskEnvironment(
    name="map_parent",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
    depends_on=[actor_env],
)


@actor_env.task
async def simple(x: int):
    print(f"[Start] Running simple with {x=}", flush=True)
    await asyncio.sleep(60)
    print("[End] simple returning", flush=True)


@actor_env.task
async def simple_with_cancel(x: int, reraise_cancel: bool = True) -> int:
    try:
        print(f"[Start] Running simple_with_cancel with {x=}", flush=True)
        await asyncio.sleep(60)
        print("[End] simple_with_cancel returning", flush=True)
    except asyncio.CancelledError:
        print("[Cancelled] simple_with_cancel cancelled, running cleanup", flush=True)
        await asyncio.sleep(1)
        print("[Cancelled] simple_with_cancel, done", flush=True)
        if reraise_cancel:
            raise

    return x * 2


@actor_env.task
async def simple_with_bad_user_code(x: int):
    try:
        print(f"[Start] Running simple_with_bad_user_code with {x=}", flush=True)
        await asyncio.sleep(60)
        print("[End] simple_with_bad_user_code returning", flush=True)
    except asyncio.CancelledError:
        print("[Cancelled] simple_with_bad_user_code was cancelled, ignoring forever", flush=True)
        while True:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                print("[Cancelled] simple_with_bad_user_code got cancelled again, ignoring", flush=True)


# aborting the parent should propagate
@parent_env.task
def parent_task():
    a = simple(x=1)
    b = simple_with_cancel(x=2)
    c = simple_with_bad_user_code(x=3)

    try:
        print("[Parent] starting gather on tasks", flush=True)
        results = asyncio.gather(a, b, c)
        print(f"[Parent] completed with {results=}", flush=True)
    except asyncio.CancelledError:
        print("[Parent] was cancelled, tasks should be cancelled too", flush=True)
        raise


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.with_runcontext().run(reuse_concurrency, n=50)
    print(run.name)
    print(run.url)
    run.wait()
    print(run.outputs())
