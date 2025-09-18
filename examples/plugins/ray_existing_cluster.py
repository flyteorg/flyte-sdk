import os
import typing

import ray

from flyte.remote import Run
from flyteplugins.ray.task import HeadNodeConfig, RayJobConfig, WorkerNodeConfig

import flyte.storage


@ray.remote
def f(x):
    return x * x


ray_config = RayJobConfig(
    head_node_config=HeadNodeConfig(ray_start_params={"log-color": "True"}),
    worker_node_config=[WorkerNodeConfig(group_name="ray-group", replicas=2)],
    enable_autoscaling=False,
    shutdown_after_job_finishes=True,
    ttl_seconds_after_finished=3600,
)

image = (
    flyte.Image.from_debian_base(name="ray")
    .with_apt_packages("wget")
    .with_pip_packages("ray[default]==2.46.0", "flyteplugins-ray")
)

task_env = flyte.TaskEnvironment(
    name="ray_client", resources=flyte.Resources(cpu=(1, 2), memory=("400Mi", "1000Mi")), image=image
)
ray_env = flyte.TaskEnvironment(
    name="ray_cluster",
    plugin_config=ray_config,
    image=image,
    resources=flyte.Resources(cpu=(2, 4), memory=("2000Mi", "4000Mi")),
)


@task_env.task()
async def hello_ray(cluster_ip: str) -> typing.List[int]:
    """
    Run a simple Ray task that connects to an existing Ray cluster.
    """
    ray.init(address=f"ray://{cluster_ip}:10001")
    futures = [f.remote(i) for i in range(5)]
    res = ray.get(futures)
    return res


@ray_env.task
async def create_ray_cluster() -> str:
    """
    Create a Ray cluster and return the head node IP address.
    """
    print("creating ray cluster")
    cluster_ip = os.getenv("MY_POD_IP")
    if cluster_ip is None:
        raise ValueError("MY_POD_IP environment variable is not set")
    return f"{cluster_ip}"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(create_ray_cluster)
    run.wait(run)
    # or get output from an existing run
    # run = Run.get("rcjt69chjnmzghzhhqjw")
    print("run url:", run.url)
    print("cluster created, running ray task")
    print("ray address:", run.outputs()[0])
    run = flyte.run(hello_ray, cluster_ip=run.outputs()[0])
    print("run url:", run.url)
