import flyte
from flyte import Image

"""
This example demonstrates how to build a new image using a private base image,


Keep in mind we only support one secret for now for a given image. If you have a private registry that you need to pull
from, and a different private registry you want to push to, and hence need more than one registry secret, this is not
supported.

Note that you do not need a secret to push to (or pull from) the default remote builder registry.
"""

image = (
    Image.from_base("ghcr.io/wild-endeavor/newprivate:0977dcd202aef2c1262ded713fe12abd")
    .clone(name="from-private-image5", registry_secret="yt_ghcr_tst5")
    .with_apt_packages("vim")
    .with_local_v2()
)

"""
Use the same registry secret in the task environment to ensure the container
can pull the private image during task execution.
"""
env = flyte.TaskEnvironment(name="private-image", image=image)


@env.task
async def t1(data: str = "hello") -> str:
    return f"Hello {data}"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(t1, data="world")
    print(run.name)
    print(run.url)
