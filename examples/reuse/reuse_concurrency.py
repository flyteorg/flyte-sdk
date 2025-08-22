import asyncio
import logging

import flyte

logger = logging.getLogger(__name__)

env = flyte.TaskEnvironment(
    name="reuse_concurrency",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    reusable=flyte.ReusePolicy(
        replicas=2,
        idle_ttl=60,
        concurrency=60,
        scaledown_ttl=60,
    ),
    image=flyte.Image.from_debian_base().with_pip_packages("unionai-reuse==0.1.4b0", pre=True),
)


@env.task
async def noop(x: int) -> int:
    logger.info(f"Task noop: {x}")
    return x


@env.task
async def reuse_concurrency(n: int = 50) -> int:
    coros = [noop(i) for i in range(n)]
    results = await asyncio.gather(*coros)
    return sum(results)


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.with_runcontext().run(reuse_concurrency, n=50)
    print(run.name)
    print(run.url)
    run.wait()
    print(run.outputs())
