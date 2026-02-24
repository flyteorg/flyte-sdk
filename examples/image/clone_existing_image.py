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
aux_env = flyte.TaskEnvironment(
    name="aux_env",
    image=image,
    resources=flyte.Resources(cpu="500m", memory="512Mi"),
)

main_env = flyte.TaskEnvironment(
    name="main_env",
    image=image,
    resources=flyte.Resources(cpu="500m", memory="512Mi"),
    depends_on=[aux_env],
)


@main_env.task
async def task_a(data: str = "hello") -> str:
    return f"a: {data}"


@aux_env.task
async def task_b(data: str = "hello") -> str:
    return f"b: {data}"


@main_env.task
async def workflow(data: str = "hello") -> str:
    """Parent task that calls a task in a different environment."""
    result_a = await task_a(data)  # same env  → cache hit  ✓
    result_b = await task_b(result_a)  # diff env  → cache miss ✗
    return result_b


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(workflow, data="hello world")
    print(run.name)
    print(run.url)
    run.wait()
