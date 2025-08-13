import asyncio
import typing

from flyte import Resources
from flyteplugins.dask import Dask, Scheduler, WorkerGroup
from distributed import Client

import flyte.remote._action
import flyte.storage


dask_plugins = "git+https://github.com/flyteorg/flyte-sdk.git@a743389f602418d7bdb572416f6f13ad8393462d#subdirectory=plugins/dask"

image = (
    flyte.Image.from_debian_base(python_version=(3, 12))
    .with_apt_packages("git")
    .with_pip_packages("dask[distributed]", dask_plugins)
)

dask_config = Dask(
    scheduler=Scheduler(
        resources=Resources(cpu="1", memory="1Gi")
    ),
    workers=WorkerGroup(
        number_of_workers=4,
        resources=Resources(cpu="1", memory="1Gi")
    )
)

task_env = flyte.TaskEnvironment(
    name="hello_dask", resources=Resources(cpu=(1, 2), memory=("400Mi", "1000Mi")), image=image
)
dask_env = flyte.TaskEnvironment(
    name="dask_env",
    plugin_config=dask_config,
    image=image,
    resources=Resources(cpu="1", memory="1Gi"),
)


@task_env.task()
async def hello_dask():
    await asyncio.sleep(20)
    print("Hello from the Dask task!")


@dask_env.task
async def hello_dask_nested(n: int = 3) -> typing.List[int]:
    print("running dask task")
    t = asyncio.create_task(hello_dask())
    client = Client()
    futures = client.map(lambda x: x + 1, range(n))
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
