from pathlib import Path

import flyte


def hello() -> str:
    return "hello"


env_cons = flyte.TaskEnvironment(
    name="albatross_envvvv",
    image=flyte.Image.from_debian_base().with_uv_project(
        pyproject_file=Path("./pyproject.toml"),
        extra_args="--only-group all",
    ),
)


@env_cons.task
async def albatross_task2() -> str:
    hello()
    print("Running albatross_task222222", flush=True)
    return f"Get bird feeder and feed"