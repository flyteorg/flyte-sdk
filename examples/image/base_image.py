from pathlib import Path

import flyte
from flyte import Image

image = (
    Image.from_debian_base(install_flyte=False)
    .with_apt_packages("vim", "wget")
    .with_pip_packages("mypy", pre=True)
    .with_dockerignore(Path(__file__).parent / ".dockerignore")
    .with_pip_packages("flyte", pre=True)
    .with_env_vars({"hello": "world16"})
    .with_pip_packages("torch")
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
