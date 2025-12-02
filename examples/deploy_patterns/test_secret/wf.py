import time
from pathlib import Path

import flyte

BUILDER_OPTIONS = {"imagepull_secret_name": "ghcr-pull-creds-02"}

sweep_base_image = (
    flyte.Image.from_debian_base(name="sweep_base_image")
    .with_env_vars({"foo": "barrrrrrrrrrrrrrrr"})
    .with_apt_packages(
        "build-essential",
        "python3-dev",
        "ssh",
        "curl",
        "git",
        "ca-certificates",
        "lsb-release",
    )
    .with_commands(
        [
            'git config --global url."https://${GITHUB_PAT}@github.com/".insteadOf "https://github.com/"',
            "env",
            "cat ~/.gitconfig",
        ],
        secret_mounts=[flyte.Secret(key="GITHUB_PAT", as_env_var="GITHUB_PAT")],
    )
    .with_env_vars(
        {
            "POETRY_VCS_REPO_USE_SYSTEM_GIT": "1",
        }
    )
    .with_poetry_project(
        pyproject_file="./pyproject.toml",
        poetry_lock=Path("./poetry.lock"),
        extra_args="--no-root",
        # Mount the secret here for use during poetry install
        secret_mounts=[flyte.Secret(key="GITHUB_PAT", as_env_var="GITHUB_PAT")],
    )
)

env = flyte.TaskEnvironment(
    name="poetry-env",
    image=sweep_base_image,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)


@env.task
def process_data(i: int) -> int:
    time.sleep(600)
    if i < 0:
        return -1
    return i + 1


if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.run(process_data, i=5)
    print(run.name)
    print(run.url)
    run.wait(run)
