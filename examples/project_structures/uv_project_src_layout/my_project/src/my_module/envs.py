import pathlib

import flyte

env = flyte.TaskEnvironment(
    name="my_module",
    image=flyte.Image.from_debian_base().with_uv_project(
        pyproject_file=pathlib.Path(__file__).parent.parent.parent / "pyproject.toml",
    ),
)