"""Same faithful uv-project repro as sh_repro_uvproject_islands.py, but on dogfood.

If this fails on dogfood too, the bug is deterministic + definition-based and
reproducible on our own cluster.
"""

import pathlib

import flyte
from flyte import Image

proj = pathlib.Path(__file__).parent / "sh_repro_uvproject"

image = Image.from_debian_base(python_version=(3, 12)).with_uv_project(
    pyproject_file=proj / "pyproject.toml",
    pre=True,
)

env = flyte.TaskEnvironment(name="sh_repro_uv", image=image)


@env.task
async def t1(data: str = "hello") -> str:
    return f"Hello {data}"


if __name__ == "__main__":
    flyte.init_from_config("/Users/ytong/.flyte/builder.remote.dogfood.staging.yaml")
    result = flyte.build(image, force=True, wait=True)
    print(f"URI: {result.uri}")
    print(f"Remote run: {result.remote_run}")
