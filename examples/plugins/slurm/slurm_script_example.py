"""Run an existing sbatch script as a Flyte task, and consume what it produces.

`slurm_script` submits a script unchanged: no container, no Flyte SDK in the image, no
edits to the script. Flyte contributes submission, phase reporting, retries, cancellation
(`scancel`), log retrieval and caching.

Inputs arrive as environment variables -- scalars as `FLYTE_INPUT_<NAME>`, `File`/`Dir` as
their URI. A script cannot write Flyte's own output format, so declared outputs work the
same way in reverse: the plugin hands the script a destination URI per output as
`FLYTE_OUTPUT_<NAME>`, the script writes there with whatever tooling the site has, and the
connector records each one once the job succeeds. The bytes go straight from the job to
object storage, never through the connector.

Connection details come from FLYTE_SLURM_HOST / FLYTE_SLURM_USERNAME /
FLYTE_SLURM_SSH_PRIVATE_KEY on the connector, or set host/username/ssh_private_key here.

    flyte run slurm_script_example.py pipeline --epochs 3
"""

import json
import pathlib

from flyteplugins.slurm import Slurm, SlurmScriptTask

import flyte
from flyte.io import File

image = flyte.Image.from_debian_base()

# An sbatch script that already works on the cluster. Only the last two lines are new:
# they copy the result to the destination Flyte provided. The upload tool is whatever the
# site has -- `aws s3 cp` here, but `rclone copyto`, `gcloud storage cp` or a Python
# client do just as well, and the job already holds credentials for reading its inputs.
SCRIPT = """#!/bin/bash
set -euo pipefail

echo "submitted from : $(hostname)"
echo "epochs         : $FLYTE_INPUT_EPOCHS"
echo "dataset        : $FLYTE_INPUT_DATASET_URI"

# Real multi-node work: the script drives srun itself, which is why script tasks are
# currently the only way to run gang-scheduled jobs through this plugin.
srun --ntasks="${SLURM_NTASKS:-1}" bash -c 'echo "rank $SLURM_PROCID on $(hostname)"'

# Stand-in for training. Write the result anywhere local, then upload it to the
# destination Flyte asked for.
python3 - <<'PYEOF' > ./summary.json
import json, os, socket
json.dump(
    {
        "epochs": int(os.environ["FLYTE_INPUT_EPOCHS"]),
        "dataset": os.environ["FLYTE_INPUT_DATASET_URI"],
        "trained_on": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID", "unset"),
    },
    open("/dev/stdout", "w"),
)
PYEOF

aws s3 cp ./summary.json "$FLYTE_OUTPUT_SUMMARY"
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
    # Declared outputs are what make this task consumable. `File` and `Dir` only: a scalar
    # would need the script and the plugin to agree on a text encoding. Several outputs
    # come back in declaration order.
    outputs={"summary": File},
    # PREEMPTED maps to RETRYABLE_FAILED, but only re-submits when retries is set.
    retries=2,
    # The cache version comes from the script body, since there is no function to hash --
    # editing the script invalidates it, changing the login node does not.
    cache="auto",
)

# A script task must belong to an environment before it can be serialized.
script_env = flyte.TaskEnvironment.from_task("slurm-script", train)

# No plugin_config, so this runs as an ordinary Kubernetes pod. The environment holding
# the task you invoke has to declare the ones its tasks call into, or their images are
# never built.
consumer_env = flyte.TaskEnvironment(
    name="slurm-script-consumer",
    image=image,
    resources=flyte.Resources(cpu="1", memory="1Gi"),
    depends_on=[script_env],
)


@consumer_env.task
async def summarize(summary: File) -> dict[str, str]:
    """Read what the script wrote, from a pod that knows nothing about Slurm.

    The output arrives as a `File`, so its bytes come from object storage. `fh.read()`
    returns a Rust-backed `Bytes`, which has no `.decode()` -- wrap it in `bytes()` first.
    """
    async with summary.open("rb") as fh:
        report = json.loads(bytes(await fh.read()).decode("utf-8"))
    return {
        "epochs": str(report["epochs"]),
        "trained_on": report["trained_on"],
        "slurm_job": report["slurm_job"],
    }


@consumer_env.task
async def pipeline(epochs: int = 3, dataset_uri: str = "gs://my-bucket/datasets/demo") -> dict[str, str]:
    """Submit the script on Slurm, then read its output back on Kubernetes."""
    summary = await train(epochs=epochs, dataset_uri=dataset_uri)
    return await summarize(summary=summary)


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(pipeline, epochs=3)
    print("run url:", run.url)
    run.wait()
    print(run.outputs())
