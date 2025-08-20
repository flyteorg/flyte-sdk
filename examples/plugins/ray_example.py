import asyncio
import typing

import ray
from flyteplugins.ray.task import HeadNodeConfig, RayJobConfig, WorkerNodeConfig

import flyte.remote
import flyte.storage


@ray.remote
def f(x):
    return x * x


ray_config = RayJobConfig(
    head_node_config=HeadNodeConfig(ray_start_params={"log-color": "True"}),
    worker_node_config=[WorkerNodeConfig(group_name="ray-group", replicas=2)],
    runtime_env={"pip": ["numpy", "pandas"]},
    enable_autoscaling=False,
    shutdown_after_job_finishes=True,
    ttl_seconds_after_finished=300,
)

image = (
    flyte.Image.from_debian_base(name="ray")
    .with_apt_packages("wget")
    .with_pip_packages("ray[default]==2.46.0", "flyteplugins-ray", "pip")
)

task_env = flyte.TaskEnvironment(
    name="hello_ray", resources=flyte.Resources(cpu=(1, 2), memory=("400Mi", "1000Mi")), image=image
)
ray_env = flyte.TaskEnvironment(
    name="ray_env",
    plugin_config=ray_config,
    image=image,
    resources=flyte.Resources(cpu=(1, 2), memory=("800Mi", "1600Mi")),
)


@task_env.task()
async def hello_ray():
    await asyncio.sleep(20)
    print("Hello from the Ray task!")


@ray_env.task
async def hello_ray_nested(n: int = 3) -> typing.List[int]:
    print("running ray task")
    t = asyncio.create_task(hello_ray())
    futures = [f.remote(i) for i in range(n)]
    res = ray.get(futures)
    await t
    return res


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(hello_ray_nested)
    print("run name:", run.name)
    print("run url:", run.url)
    run.wait(run)

    action_details = flyte.remote.ActionDetails.get(run_name=run.name, name="a0")
    for log in action_details.pb2.attempts[-1].log_info:
        print(f"{log.name}: {log.uri}")
