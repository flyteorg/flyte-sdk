import asyncio

import flyte

env = flyte.TaskEnvironment(
    name="large_fanout_concurrent",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    reusable=flyte.ReusePolicy(
        replicas=10,
        idle_ttl=60,
        concurrency=5,
        scaledown_ttl=60,
    ),
    image=flyte.Image.from_debian_base().with_pip_packages("unionai-reuse==0.1.4"),
)


@env.task
async def noop(x: int) -> int:
    await asyncio.sleep(1)
    return x


@env.task
async def reuse_concurrency(n: int = 50) -> int:
    coros = [noop(i) for i in range(n)]
    results = await asyncio.gather(*coros)
    return sum(results)


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.with_runcontext().run(reuse_concurrency, n=1000)
    print(run.name)
    print(run.url)
    run.wait()
    print(run.outputs())
