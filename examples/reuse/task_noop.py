import logging

import flyte

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

noop_env = flyte.TaskEnvironment(
    name="reuse_concurrency",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    reusable=flyte.ReusePolicy(
        replicas=2,
        idle_ttl=60,
        concurrency=100,
        scaledown_ttl=60,
    ),
    image=flyte.Image.from_name("abc"),
    # image=flyte.Image.from_debian_base().with_pip_packages("unionai-reuse==0.1.5"),
)


@noop_env.task
async def noop(x: int) -> int:
    logger.debug(f"Task noop: {x}")
    return x
