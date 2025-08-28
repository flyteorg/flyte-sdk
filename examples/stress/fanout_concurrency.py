import asyncio

import flyte

env = flyte.TaskEnvironment(
    name="large_fanout_concurrent",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    reusable=flyte.ReusePolicy(
        replicas=10,
        idle_ttl=60,
        concurrency=50,
        scaledown_ttl=60,
    ),
    image=flyte.Image.from_debian_base().with_pip_packages("unionai-reuse==0.1.4"),
)


@env.task
async def noop(x: int) -> int:
    return x


@env.task
async def reuse_concurrency(n: int = 50) -> int:
    coros = [noop(i) for i in range(n)]
    results = await asyncio.gather(*coros)
    return sum(results)


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    runs = []
    for i in range(10):
        run = flyte.run(reuse_concurrency, n=1000)
        runs.append(run.url)
    print(runs)
