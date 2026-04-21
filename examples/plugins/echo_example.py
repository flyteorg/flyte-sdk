import pathlib

from flyteplugins.echo import Echo

import flyte

# The echo plugin runs in leaseworker, not inside this task image.
# Use any normal image your environment can resolve.
image = flyte.Image.from_debian_base(python_version=(3, 12))

echo_env = flyte.TaskEnvironment(
    name="echo-env",
    image=image,
    plugin_config=Echo(),
)


@echo_env.task
async def echo_identity(message: str) -> str:
    return message


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(echo_identity, message="hello")
    print("run name:", run.name)
    print("run url:", run.url)
