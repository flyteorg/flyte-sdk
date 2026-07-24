"""Reusable shared Ray cluster.

A driver task spawns two Ray tasks concurrently. Because both Ray tasks belong to the same
environment declared with ``flyte.ReusePolicy``, they run on ONE long-lived, shared RayCluster
instead of each paying a full cluster cold-start: the first task to arrive creates the cluster,
the second reuses it. The cluster is shut down automatically once it has been idle for ``idle_ttl``
seconds.
"""

import asyncio
import typing

import ray
from flyteplugins.ray.task import HeadNodeConfig, RayJobConfig, WorkerNodeConfig

import flyte


@ray.remote
def square(x: int) -> int:
    return x * x


ray_config = RayJobConfig(
    head_node_config=HeadNodeConfig(),
    worker_node_config=[WorkerNodeConfig(group_name="ray-group", replicas=2)],
    enable_autoscaling=False,
)

image = (
    flyte.Image.from_debian_base(name="flyte")
    .with_apt_packages("wget")
    .with_pip_packages("ray[default]==2.46.0", "flyteplugins-ray")
)

# The reusable Ray environment. Every task in this environment with an identical configuration
# shares one long-lived RayCluster (keyed by the environment's identity). `idle_ttl` shuts the
# cluster down after it sits idle with no jobs for that many seconds.
ray_env = flyte.TaskEnvironment(
    name="reusable_ray_env",
    plugin_config=ray_config,
    image=image,
    resources=flyte.Resources(cpu=(1, 2), memory=("2000Mi", "4000Mi")),
    reusable=flyte.ReusePolicy(replicas=1, idle_ttl=600),
)

# A plain environment for the driver that orchestrates the Ray tasks. The driver itself does not
# need Ray — it just fans out to the Ray tasks. `depends_on=[ray_env]` deploys the Ray environment
# alongside the driver so the driver can spawn tasks in it.
driver_env = flyte.TaskEnvironment(name="flyte_driver_env", image=image, depends_on=[ray_env])


@ray_env.task
async def sum_of_squares(n: int) -> int:
    """Runs on the shared RayCluster and fans the work out to Ray workers via `@ray.remote`..."""
    print(f"running Ray task on the shared cluster (n={n})")
    results = ray.get([square.remote(i) for i in range(n)])
    print(f"partial results: {results}")
    return sum(results)


@driver_env.task
async def driver(a: int = 5, b: int = 8) -> typing.List[int]:
    """Spawns two Ray tasks concurrently; both bind to the same reusable RayCluster.

    The first task to reach the cluster creates it; the second reuses it (no second cold start).
    """
    first, second = await asyncio.gather(
        sum_of_squares(a),
        sum_of_squares(b),
    )
    print(f"driver results: sum_of_squares({a})={first}, sum_of_squares({b})={second}")
    return [first, second]


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(driver, a=5, b=8)
    print("run url:", run.url)
    run.wait()
    print("phase:", run.action.phase if run.action else "unknown")
