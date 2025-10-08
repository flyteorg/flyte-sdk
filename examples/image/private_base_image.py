from pathlib import Path

import flyte
from flyte import Image

image = (
    Image.from_base("pingsutw/private:d1742efed83fc4b18c7751e53e771bbe")
    .clone(registry="registry-1.docker.io/pingsutw", name="private", secret="pingsutw")
    # Image.from_debian_base(install_flyte=False, registry="registry-1.docker.io/pingsutw", name="private", secret="pingsutw")
    .with_apt_packages("vim", "wget")
    .with_pip_packages("ty", pre=True)
    .with_env_vars({"hello": "world6"})
    .with_dockerignore(Path(__file__).parent / ".dockerignore")
    .with_local_v2()
)

env = flyte.TaskEnvironment(name="t1", image=image, secrets=[flyte.Secret(key="pingsutw")])


@env.task
async def t1(data: str = "hello") -> str:
    return f"Hello {data}"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(t1, data="world")
    print(run.name)
    print(run.url)
