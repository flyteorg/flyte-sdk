import flyte
from flyte import Image

"""
This example demonstrates how to build a new image using a private base image,

Keep in mind we only support one secret for now for a given image. If you have a private registry that you need to pull
from, and a different private registry you want to push to, and hence need more than one registry secret, this is not
supported.

Note that you do not need a secret to push to (or pull from) the default remote builder registry.

Please see `flyte create secret --help` for all the options.
"""

BASE_IMG_URL = "ghcr.io/username/private:123123"

"""
In this first example, set the registry_secret on the image, so that when building the image, the remote builder
can pull the base image from the private registry, and build from there. However since the registry that the remote
builder pushes to is already availble for tasks, you don't need to specify a secret in the task environment itself.
"""
image = (
    Image.from_base(BASE_IMG_URL)
    .clone(name="from-private-image5", registry_secret="my_registry_secret")
    .with_apt_packages("vim")
    .with_local_v2()
)

env = flyte.TaskEnvironment(name="private-image", image=image)


@env.task
async def t1(data: str = "hello") -> str:
    return f"Hello {data}"


# For this second example, we only set the secret on the task environment, and there is no build step.
# When running the task, we need the secret to pull the base image to run the task.
env_base_only = flyte.TaskEnvironment(
    name="private-image", image=flyte.Image.from_base(BASE_IMG_URL), secrets="my_registry_secret"
)


@env_base_only.task
async def t2(data: str = "hello") -> str:
    return f"Hello {data}"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(t1, data="world")
    print(run.name)
    print(run.url)
