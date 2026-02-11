import asyncio

import flyte
from flyte import Image

# Start from an existing base image and clone it so the default configured builder picks it up.
# The .clone() call assigns a name, which triggers a build.
image = Image.from_base("ghcr.io/flyteorg/flyte:py3.12-v2.0.0b56").clone(
    name="my-flyte-image",
)

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
