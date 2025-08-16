"""Example of how to parameterizing images with environment variables.

First build an image:

```
flyte build image_from_env.py build_env
``

Make sure the built image is publicly accessible.

Then take the image uri and export it as an environment variable:

```
export BASE_IMAGE=...
```

Then run the task:

```
python image_from_env.py
```
"""

import os
import flyte


# Task environment for building the image
build_env = flyte.TaskEnvironment(
    name="build_env",
    image=(
        flyte.Image
        .from_debian_base(name="base-image", python_version=(3, 12))
        .with_pip_packages("flyte", pre=True, extra_args="--prerelease=allow")
    ),
)

image_env_var = "BASE_IMAGE"
image_uri = os.environ[image_env_var]

# This task environment uses the BASE_IMAGE environment variable to set the image.
env = flyte.TaskEnvironment(
    name="image_from_env",
    env={image_env_var: image_uri},
    image=flyte.Image.from_base(image_uri),
)


@env.task
async def t1(data: str = "hello") -> str:
    return f"Hello {data}"

@env.task
async def main(data: str = "hello") -> str:
    return await t1(data)


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(main, data="hello world")
    print(run.name)
    print(run.url)
