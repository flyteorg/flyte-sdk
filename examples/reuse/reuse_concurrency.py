import asyncio
import logging

import flyte
from pathlib import Path
from flyte._image import PythonWheels

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

actor_dist_folder = Path("/Users/ytong/go/src/github.com/unionai/flyte/fasttask/worker-v2/dist")
wheel_layer = PythonWheels(wheel_dir=actor_dist_folder, package_name="unionai-reuse")
base = flyte.Image.from_debian_base()
actor_image = base.clone(addl_layer=wheel_layer)


env = flyte.TaskEnvironment(
    name="reuse_concurrency",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    reusable=flyte.ReusePolicy(
        replicas=2,
        idle_ttl=60,
        concurrency=100,
        scaledown_ttl=60,
    ),
    # image=flyte.Image.from_debian_base().with_pip_packages("unionai-reuse==0.1.7"),
    image=actor_image,
)


@env.task
async def noop(x: int) -> int:
    logger.warning(f"Task noop: {x}")
    return x


@env.task
async def reuse_concurrency(n: int = 50) -> int:
    coros = [noop(i) for i in range(n)]
    results = await asyncio.gather(*coros)
    return sum(results)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext().run(reuse_concurrency, n=500)
    print(run.name)
    print(run.url)
    run.wait()
