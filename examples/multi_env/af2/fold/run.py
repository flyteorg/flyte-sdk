import flyte
from flyte.io import File

FOLD_PACKAGES = ["ruff"]

fold_image = flyte.Image.from_debian_base().with_pip_packages(*FOLD_PACKAGES)

env = flyte.TaskEnvironment(name="multi_fold", image=fold_image)


@env.task
def run_fold(sequence: str, msa: File) -> list[str]:
    with msa.open_sync("r") as f:
        msa_content = f.read()
    return [msa_content, sequence]
