import logging
import pathlib

from fold.run import env as fold_env
from fold.run import fold_image, run_fold
from msa.run import MSA_PACKAGES, run_msa
from msa.run import env as msa_env

import flyte

af2_image = fold_image.with_pip_packages(*MSA_PACKAGES)
env = flyte.TaskEnvironment(
    name="multi_env",
    depends_on=[fold_env, msa_env],
    image=af2_image,
)


@env.task
def run_af2(sequence: str) -> list[str]:
    """Given string, output files containing protein structure
    This requires model weights + gpus + large database on aws fsx lustre
    """
    print(f"Running AlphaFold2 for sequence: {sequence}")
    msa = run_msa(sequence)
    print(f"MSA result: {msa}, passing to fold task")
    results = run_fold(sequence, msa)
    print(f"Fold results: {results}")
    return results


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent, log_level=logging.INFO)
    r = flyte.run(run_af2, "AAGGTTCCAA")
    print(r.url)
    # print(r.outputs())
