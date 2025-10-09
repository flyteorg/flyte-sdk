import flyte
from flyte import Image

"""
This example demonstrates how to build a new image using a private base image,
then push the resulting image back to the same private registry.

To create a registry secret, run:
```
flyte create secret --type image_pull pingsutw --from-file <PATH>/docker/config.json
```
"""
image = (
    Image.from_base("pingsutw/private:d1742efed83fc4b18c7751e53e771bbe")
    .clone(registry="docker.io/pingsutw", name="private", registry_secret="pingsutw")
    .with_apt_packages("vim")
    .with_local_v2()
)

"""
Use the same registry secret in the task environment to ensure the container
can pull the private image during task execution.
"""
env = flyte.TaskEnvironment(name="private-image", image=image, secrets="pingsutw")


@env.task
async def t1(data: str = "hello") -> str:
    return f"Hello {data}"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(t1, data="world")
    print(run.name)
    print(run.url)
