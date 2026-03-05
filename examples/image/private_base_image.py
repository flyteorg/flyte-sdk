import flyte
from flyte import Image

"""
This example demonstrates how to build a new image using a private base image,
then push the resulting image back to the same private registry.

This workflow will change in the next couple weeks, but for now:

First run the old CLI command to create a file. This will extract out from your local docker daemon logged in tokens.
```
union create imagepullsecret
```


Take the file that was created and then call the following to create a registry secret:
```
flyte create secret --type image_pull pingsutw --from-file <PATH/to/above file>
```

Keep in mind we only support one secret for now for a given image. If you have a private registry that you need to pull
from, and a different private registry you want to push to, and hence need more than one registry secret, this is not
supported.

Note that you do not need a secret to push to (or pull from) the default remote builder registry. So in the example
below, if you drop the `.clone(...)` call, you can also drop the `secrets` argument in the TaskEnvironment.

"""
image = (
    Image.from_base("pingsutw/private:d1742efed83fc4b18c7751e53e771bbe")
    .clone(registry="docker.io/pingsutw", name="private", registry_secret="pingsutw", extendable=True)
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
