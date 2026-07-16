"""E2E test for the fastray (reusable shared RayCluster) path on dogfood-gcp."""

import asyncio
import typing

import ray
from flyteplugins.ray.task import HeadNodeConfig, RayJobConfig, WorkerNodeConfig

import flyte


@ray.remote
def f(x):
    return x * x


ray_config = RayJobConfig(
    head_node_config=HeadNodeConfig(),
    worker_node_config=[WorkerNodeConfig(group_name="ray-group", replicas=2)],
    enable_autoscaling=False,
    shutdown_after_job_finishes=True,
    ttl_seconds_after_finished=300,
)

image = (
    flyte.Image.from_debian_base(name="ray")
    .with_apt_packages("wget")
    .with_pip_packages("ray[default]==2.46.0", "flyteplugins-ray", "pip", "mypy")
)

ray_env = flyte.TaskEnvironment(
    name="fastray_env",
    plugin_config=ray_config,
    image=image,
    resources=flyte.Resources(cpu=(1, 2), memory=("2000Mi", "4000Mi")),
    reusable=flyte.ReusePolicy(replicas=1, idle_ttl=600),
)


@ray_env.task
async def hello_fastray(n: int = 3) -> typing.List[int]:
    print("running fastray task")
    futures = [f.remote(i) for i in range(n)]
    res = ray.get(futures)
    print(f"results: {res}")
    return res


if __name__ == "__main__":
    flyte.init_from_config("/Users/kevin/.flyte/config-dogfood-gcp.yaml", project="flytesnacks", domain="development")
    run = flyte.run(hello_fastray, n=3)
    print("run url:", run.url)
    run.wait()
    print("phase:", run.action.phase if run.action else "unknown")
