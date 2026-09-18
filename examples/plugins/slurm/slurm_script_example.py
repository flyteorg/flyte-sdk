"""Run an existing sbatch script as a Flyte task.

`slurm_script` submits a script unchanged: no container, no Flyte SDK in the image, no
edits to the script. Flyte contributes submission, phase reporting, retries, cancellation
(`scancel`) and log retrieval.

Scalar inputs arrive as `FLYTE_INPUT_<NAME>`. Non-scalars are silently dropped -- pass a
URI as a `str` and let the script fetch it.

There are no typed outputs, so nothing downstream can consume a script task's results
through Flyte. Use `slurm_example.py` when you need that.

Connection details come from FLYTE_SLURM_HOST / FLYTE_SLURM_USERNAME /
FLYTE_SLURM_SSH_PRIVATE_KEY on the connector, or set host/username/ssh_private_key here.

    flyte run --local slurm_script_example.py train
"""

import pathlib

import flyte
from flyteplugins.slurm import Slurm, SlurmScriptTask

# An sbatch script that already works on the cluster. Our #SBATCH directives are emitted
# first, so its own are kept but lose where they conflict; a leading shebang is dropped.
SCRIPT = """#!/bin/bash
set -euo pipefail

echo "submitted from : $(hostname)"
echo "epochs         : $FLYTE_INPUT_EPOCHS"
echo "dataset        : $FLYTE_INPUT_DATASET_URI"

# Real multi-node work: the script drives srun itself, which is why script tasks are
# currently the only way to run gang-scheduled jobs through this plugin.
srun --ntasks="${SLURM_NTASKS:-1}" bash -c 'echo "rank $SLURM_PROCID on $(hostname)"'
"""

train = SlurmScriptTask(
    name="train",
    script=SCRIPT,
    plugin_config=Slurm(
        partition="main",
        nodes=1,
        time_limit="0:10:00",
    ),
    inputs={"epochs": int, "dataset_uri": str},
    retries=2,  # PREEMPTED maps to RETRYABLE_FAILED, so preemption is retried
)

# A task must belong to an environment before it can be serialized.
env = flyte.TaskEnvironment.from_task("slurm-script", train)


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(train, epochs=3, dataset_uri="gs://my-bucket/datasets/demo/v1")
    print("run url:", run.url)
    run.wait()
