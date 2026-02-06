import asyncio

import flyte

env = flyte.TaskEnvironment(
    name="fanout",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    reusable=flyte.ReusePolicy(
        replicas=10,
        idle_ttl=60,
        concurrency=100,
        scaledown_ttl=60,
    ),
    image=flyte.Image.from_debian_base().with_pip_packages("unionai-reuse"),
)


@env.task
async def worker(x: int) -> int:
    if x % 1000 == 0:
        print(f"Simulating failure for input: {x}")
        raise ValueError("Simulated failure")
    else:
        print(f"Processing input: {x}")

    return x


@env.task
async def fanout_with_failures(n: int = 50) -> int:
    with flyte.group("fanout_with_failures"):
        coros = [worker(i) for i in range(n)]
        results = await asyncio.gather(*coros, return_exceptions=True)
        total = sum(r for r in results if not isinstance(r, Exception))
        return total


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext().run(fanout_with_failures, n=500)
    print(run.name)
    print(run.url)
