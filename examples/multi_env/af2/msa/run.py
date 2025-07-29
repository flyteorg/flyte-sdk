import flyte
from flyte.io import File

MSA_PACKAGES = ["pytest"]

msa_image = flyte.Image.from_debian_base().with_pip_packages(*MSA_PACKAGES)

env = flyte.TaskEnvironment(name="multi_msa", image=msa_image)


@env.task
def run_msa(x: str) -> File:
    f = File.new_remote()
    with f.open_sync("w") as fp:
        fp.write(x)
    return f
