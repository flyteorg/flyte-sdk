import asyncio

import flyte
from flyte._image import PythonWheels
from pathlib import Path

controller_dist_folder = Path("/Users/ytong/go/src/github.com/flyteorg/sdk-rust/rs_controller/dist")
wheel_layer = PythonWheels(wheel_dir=controller_dist_folder, package_name="flyte_controller_base")
base = flyte.Image.from_debian_base()
rs_controller_image = base.clone(addl_layer=wheel_layer)

env = flyte.TaskEnvironment("large-slow-run", image=rs_controller_image,)


@env.task
async def sleeper():
    print("I am going for my first sleep!!!!!!!", flush=True)
    await asyncio.sleep(1)
    print("I am going for my second sleep!!!!!!!", flush=True)
    await asyncio.sleep(1)
    print("I am done....", flush=True)


@env.task
async def parallel_main(n: int):
    coros = []
    for i in range(n):
        coros.append(sleeper())
    await asyncio.gather(*coros)


@env.task
async def main(n: int):
    for i in range(n):
        await sleeper()
