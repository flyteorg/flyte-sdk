import flyte
from flyte import Image, Secret

private_package = "git+https://$GITHUB_PAT@github.com/pingsutw/flytex.git@2e20a2acebfc3877d84af643fdd768edea41d533"
image = (
    Image.from_debian_base(install_flyte=True)
    .with_apt_packages("git")
    .with_pip_packages("mypy", private_package, pre=True, secret_mounts=Secret("GITHUB_PAT"))
    .with_commands(["env"], secret_mounts=Secret("GITHUB_PAT"))
    .with_commands(["sh -c 'echo ${GITHUB_PAT}'"], secret_mounts=Secret("GITHUB_PAT"))
    .with_commands(["echo ${GITHUB_PAT}"], secret_mounts=Secret("GITHUB_PAT"))
    .with_commands(["echo $GITHUB_PAT"], secret_mounts=Secret("GITHUB_PAT"))
    .with_commands(["ls /etc/flyte/secrets"], secret_mounts=Secret("GITHUB_PAT", mount="/etc/flyte/secrets"))
)

env = flyte.TaskEnvironment(name="private_package", image=image)


@env.task
async def t1(data: str = "hello") -> str:
    return f"Hello {data}"


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(t1, data="world")
    print(run.name)
    print(run.url)
