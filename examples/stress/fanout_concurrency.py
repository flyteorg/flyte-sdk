import asyncio

import flyte
from pathlib import Path
from flyte._image import PythonWheels

actor_dist_folder = Path("/Users/ytong/go/src/github.com/unionai/flyte/fasttask/worker-v2/dist")
wheel_layer = PythonWheels(wheel_dir=actor_dist_folder, package_name="unionai-reuse")
base = flyte.Image.from_debian_base()
actor_image = base.clone(addl_layer=wheel_layer)

env = flyte.TaskEnvironment(
    name="large_fanout_concurrent",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    reusable=flyte.ReusePolicy(
        replicas=10,
        idle_ttl=60,
        concurrency=50,
        scaledown_ttl=60,
    ),
    image=actor_image,
)


@env.task
async def noop(x: int) -> int:
    return x


@env.clone_with(name="fanout_main", reusable=None, depends_on=[env]).task
async def reuse_concurrency(n: int = 50) -> int:
    coros = [noop(i) for i in range(n)]
    results = await asyncio.gather(*coros)
    return sum(results)


if __name__ == "__main__":
    flyte.init_from_config()
    runs = []
    for i in range(1):
        run = flyte.run(reuse_concurrency, n=1000)
        runs.append(run.url)
    print(runs)
