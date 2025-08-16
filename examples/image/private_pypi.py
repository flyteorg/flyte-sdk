from pathlib import Path

import flyte
from flyte import Image

image = (
    Image.from_debian_base(install_flyte=True)
    .with_pip_packages("mypy", pre=True)
)

env = flyte.TaskEnvironment(name="private_package", image=image)


@env.task
async def t1(data: str = "hello") -> str:
    import flytex
    return f"Hello {data}"


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(t1, data="world")
    print(run.name)
    print(run.url)
