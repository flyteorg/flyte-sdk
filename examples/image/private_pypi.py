import flyte
from flyte import Image, Secret

private_package = "git+https://$GITHUB_PAT@github.com/pingsutw/flytex.git@2e20a2acebfc3877d84af643fdd768edea41d533"
image = (
    Image.from_debian_base(install_flyte=True)
    .with_apt_packages("git", "vim", "curl")
    .with_pip_packages(private_package, pre=True, secret_mounts=Secret("GITHUB_PAT"))
)

env = flyte.TaskEnvironment(name="private_package", image=image)


@env.task
async def t1(data: str = "hello") -> str:
    return f"Hello {data}"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(t1, data="world")
    print(run.name)
    print(run.url)
