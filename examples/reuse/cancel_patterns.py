import asyncio

import flyte

actor_env = flyte.TaskEnvironment(
    name="reuse_cancel_patterns",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    reusable=flyte.ReusePolicy(
        replicas=1,
        idle_ttl=60,
        concurrency=5,
        scaledown_ttl=60,
    ),
    image=flyte.Image.from_debian_base().with_pip_packages("unionai-reuse==0.1.4b0", pre=True),
)

parent_env = flyte.TaskEnvironment(
    name="map_parent",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
    depends_on=[actor_env],
)



@actor_env.task
async def simple(x: int) -> int:
    print(f"[Start] Running simple with {x=}", flush=True)
    await asyncio.sleep(20)
    print(f"[End] simple returning", flush=True)
    return x


@actor_env.task
async def simple_with_cancel(x: int) -> int:
    try:
        print(f"[Start] Running simple_with_cancel with {x=}", flush=True)
        await asyncio.sleep(20)
        print(f"[End] simple_with_cancel returning", flush=True)
    except asyncio.CancelledError:
        print(f"[Cancelled] simple_with_cancel was cancelled", flush=True)
        raise


@actor_env.task
async def simple_with_long_cancel(x: int) -> int:
    try:
        print(f"[Start] Running simple_with_long_cancel with {x=}", flush=True)
        await asyncio.sleep(20)
        print(f"[End] simple_with_long_cancel returning", flush=True)
    except asyncio.CancelledError:
        print(f"[Cancelled] simple_with_long_cancel was cancelled", flush=True)
        await asyncio.sleep(10)
        print(f"[Cancel Handled] simple_with_long_cancel finished handling", flush=True)
        raise


# aborting the parent should propagate
@parent_env.task
def parent_task():
    a = simple(x=1)
    b = simple_with_cancel(x=2)
    c = simple_with_long_cancel(x=3)

    try:
        print(f"[Parent] starting gather on tasks", flush=True)
        results = asyncio.gather(a, b, c)
        print(f"[Parent] completed with {results=}", flush=True)
    except asyncio.CancelledError:
        print(f"[Parent] was cancelled, tasks should be cancelled too", flush=True)
        raise


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.with_runcontext().run(reuse_concurrency, n=50)
    print(run.name)
    print(run.url)
    run.wait()
    print(run.outputs())
