import asyncio

import flyte
from flyte import Image

# Start from an existing base image and clone it so the default configured builder picks it up.
# The .clone() call assigns a name, which triggers a build.
# By default, images are NOT extendable (extendable=False), meaning you cannot add layers on top of them.
# To add layers, you must explicitly set extendable=True.
image = Image.from_base("ghcr.io/flyteorg/flyte:py3.12-v2.0.0b56").clone(
    name="my-flyte-image",
)

# To add layers to an image, you must set extendable=True:
# extendable_image = image.clone(name="my-extendable-image", extendable=True)
# image_with_packages = extendable_image.with_pip_packages("requests", "pandas")

# Without extendable=True, this will raise an error:
# image.with_pip_packages("numpy")  # ValueError: Cannot add additional layers to a non-extendable image

env = flyte.TaskEnvironment(
    name="my-flyte-task",
    image=image,
    resources=flyte.Resources(cpu="500m", memory="512Mi"),
)


@env.task
async def sleep_task(seconds: int = 10) -> str:
    print(f"Sleeping for {seconds} seconds...", flush=True)
    await asyncio.sleep(seconds)
    print("Done!", flush=True)
    return f"Slept for {seconds} seconds"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(sleep_task, seconds=10)
    print(run.name)
    print(run.url)
    run.wait()
