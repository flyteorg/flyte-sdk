from pathlib import Path

import flyte
from flyte import Image

image = (
    Image.from_debian_base(install_flyte=False)
    .with_uv_project(uvlock=Path("../../uv.lock"), pyproject_file=Path("../../pyproject.toml"))
    .with_local_v2()
)

env = flyte.TaskEnvironment(name="t1", image=image)


@env.task
async def t1(data: str = "hello") -> str:
    return f"Hello {data}"


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(t1, data="world")
    print(run.name)
    print(run.url)
