from pathlib import Path

import flyte
from flyte import Image

image1 = Image.from_debian_base(name="base-image", python_version=(3, 12)).with_pip_packages(
    "flyte", "ty", pre=True, extra_args="--prerelease=allow"
)
image2 = image1.with_apt_packages("vim")
image3 = image2.with_commands(["pwd", "ls -al"]).with_requirements(Path(__file__).parent / "requirements.txt")
image4 = image3.with_source_file(Path(__file__).parent.parent.parent / "Makefile").with_env_vars({"hello": "world3"})

env1 = flyte.TaskEnvironment(name="t1", image=image1)
env2 = flyte.TaskEnvironment(name="t2", image=image2, depends_on=[env1])
env3 = flyte.TaskEnvironment(name="t3", image=image3, depends_on=[env2])
env4 = flyte.TaskEnvironment(name="t4", image=image4, depends_on=[env3])


@env4.task
async def t4(data: str = "hello") -> str:
    return f"Hello {data}"


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    run = flyte.run(t4, data="hello world")
    print(run.name)
    print(run.url)
    run.wait(run)
