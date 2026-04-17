import pathlib
from datetime import timedelta

import flyte
from flyte.extras import Sleep

# The core-sleep plugin executes in leaseworker, so no task pod is created.
image = flyte.Image.from_debian_base(python_version=(3, 12))

sleep_env = flyte.TaskEnvironment(
    name="sleep-env",
    image=image,
    plugin_config=Sleep(),
)


@sleep_env.task
async def sleep_for(duration: timedelta) -> None:
    return None


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(sleep_for, duration=timedelta(seconds=60))
    print("run name:", run.name)
    print("run url:", run.url)
