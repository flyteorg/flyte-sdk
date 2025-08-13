import asyncio
import typing
from pathlib import Path

from flyte import Resources
from flyteplugins.dask import Dask, Scheduler, WorkerGroup

import flyte.remote._action
import flyte.storage


def inc(x):
    return x + 1

dask_plugins = f"git+https://github.com/flyteorg/flytekit.git@80beb008094d5c7fa4126b33d9bc7ec67c2b8551#subdirectory=plugins/dask"

image = (
    flyte.Image.from_debian_base(python_version=(3, 12))
    .with_apt_packages("git")
    .with_pip_packages("dask[distributed]==2024.12.1", dask_plugins)
    .with_source_folder(Path(__file__).parent.parent.parent / "plugins/dask", "./dask")
    .with_env_vars({"PYTHONPATH": "./dask/src:${PYTHONPATH}", "hello": "world1"})
)

dask_config = Dask(
    scheduler=Scheduler(
        resources=Resources(cpu=(1, 2), memory=("800Mi", "1600Mi"))
    ),
    workers=WorkerGroup(
        number_of_workers=4,
        resources=Resources(cpu="1", memory="1Gi")
    )
)

task_env = flyte.TaskEnvironment(
    name="hello_dask", resources=Resources(cpu=(1, 2), memory=("400Mi", "1000Mi")), image=image
)
ray_env = flyte.TaskEnvironment(
    name="ray_env",
    plugin_config=dask_config,
    image=image,
    resources=Resources(cpu=(1, 2), memory=("800Mi", "1600Mi")),
)


@task_env.task()
async def hello_dask():
    await asyncio.sleep(20)
    print("Hello from the Dask task!")


@ray_env.task
async def hello_dask_nested(n: int = 3) -> typing.List[int]:
    from distributed import Client

    print("running dask task")
    t = asyncio.create_task(hello_dask())
    client = Client()
    futures = client.map(inc, range(n))
    res = client.gather(futures)
    await t
    return res


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(hello_dask_nested)
    print("run name:", run.name)
    print("run url:", run.url)
    run.wait(run)

    action_details = flyte.remote._action.ActionDetails.get(run_name=run.name, name="a0")
    for log in action_details.pb2.attempts[-1].log_info:
        print(f"{log.name}: {log.uri}")
