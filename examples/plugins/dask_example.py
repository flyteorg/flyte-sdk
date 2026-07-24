import asyncio
import pathlib
import typing

from distributed import Client
from flyteplugins.dask import Dask, Scheduler, WorkerGroup

import flyte.remote
import flyte.storage
from flyte import Resources

# The scheduler and worker pods inherit this image (Scheduler()/WorkerGroup() below leave
# image unset), and the Dask K8s operator launches them via the standalone `dask-scheduler`
# and `dask-worker` commands. Those console scripts were removed from `distributed` in
# 2026.6.0 (replaced by the `dask scheduler` / `dask worker` subcommands), so a newer
# dask[distributed] makes the cluster pods crash-loop with
# "dask-scheduler: executable file not found in $PATH". Pin below 2026.6.0 to keep them.
# (flyteplugins-dask caps this itself as of the next release; the explicit pin here keeps
# the example working against the currently-published, un-capped plugin.)
#
# connectrpc<0.11 keeps the runtime from hitting "'Headers' object is not callable":
# connectrpc 0.11 turned RequestContext.request_headers into a property, which the flyte
# auth interceptor (method-call style) can't use. flyte>=2.5.10 caps this too, but pinning
# here guarantees a clean rebuild picks up a compatible connectrpc.
image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "flyteplugins-dask",
    "dask[distributed]<2026.6.0",
    "connectrpc<0.11",
    "bokeh",
)

dask_config = Dask(
    scheduler=Scheduler(),
    workers=WorkerGroup(number_of_workers=4),
)

task_env = flyte.TaskEnvironment(
    name="hello_dask", resources=Resources(cpu=(1, 2), memory=("400Mi", "1000Mi")), image=image
)
dask_env = flyte.TaskEnvironment(
    name="dask_env",
    plugin_config=dask_config,
    image=image,
    resources=Resources(cpu="1", memory="1Gi"),
    depends_on=[task_env],
)


@task_env.task()
async def hello_dask():
    await asyncio.sleep(5)
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
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(hello_dask_nested, n=3)
    print("run name:", run.name)
    print("run url:", run.url)
    run.wait()

    action_details = flyte.remote.ActionDetails.get(run_name=run.name, name="a0")
    for log in action_details.pb2.attempts[-1].log_info:
        print(f"{log.name}: {log.uri}")
