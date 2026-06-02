"""Faithful repro of the customer's failing definition, against islands.

Mirrors their image: from_debian_base(3.12) (install_flyte defaults to True) +
with_uv_project(..., pre=True). The dummy pyproject/uv.lock live in
./sh_repro_uvproject/. Single build, one at a time.
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
    flyte.init_from_config(
        "/Users/ytong/go/src/github.com/unionai/cloud/gen/cli-config/uctl/islands.production_v2.yaml"
    )
    result = flyte.build(image, force=True, wait=True)
    print(f"URI: {result.uri}")
    print(f"Remote run: {result.remote_run}")
