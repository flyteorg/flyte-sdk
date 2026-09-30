"""Run an existing sbatch script as a Flyte task, and consume what it produces.

`slurm_script` submits a script unchanged: no container, no Flyte SDK in the image, no
edits to the script. Flyte contributes submission, phase reporting, retries, cancellation
(`scancel`), log retrieval and caching.

Inputs arrive as environment variables -- scalars as `FLYTE_INPUT_<NAME>`, `File`/`Dir` as
their URI. A script cannot write Flyte's own output format, so declared outputs work the
same way in reverse: the plugin hands the script a destination per output as
`FLYTE_OUTPUT_<NAME>` and records each one once the job succeeds.

By default that destination is a **local path** and the connector streams the file to
object storage afterwards, so the compute node needs no upload tool and no credentials --
which is what this example uses. Pass `output_upload="job"` and the destination becomes the
object-storage URI for the script to upload to directly; that is the right choice for
anything large, since the connector refuses to carry more than 100 MB.

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

# An sbatch script that already works on the cluster. Only the last line is new: it copies
# the result to the destination Flyte provided, which with the default `output_upload` is an
# ordinary local path -- hence `cp` rather than an object-storage client. The connector
# moves it from there.
SCRIPT = """#!/bin/bash
set -euo pipefail

echo "submitted from : $(hostname)"
echo "epochs         : $FLYTE_INPUT_EPOCHS"
echo "dataset        : $FLYTE_INPUT_DATASET_URI"

# The script drives srun itself, which is why script tasks are currently the only way to
# run gang-scheduled work through this plugin: raise `nodes` below and every rank in the
# allocation runs here. A native task is pinned to one task on one node.
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

# The destination is a local path under ~/.flyte/jobs/<job>.outputs (the plugin creates
# the directory), so this is a plain copy. Switching to output_upload="job" makes it the
# `s3://`/`gs://` URI, and then `cp` cannot write it -- use the store's client, e.g.
#   GCS=":gcs,service_account_file=$HOME/.gcp/sa.json,bucket_policy_only=true:"
#   rclone copyto ./summary.json "${GCS}${FLYTE_OUTPUT_SUMMARY#gs://}"
# Both lines have to change together; leaving `cp` in place fails with
# "cp: cannot create regular file 'gs://...': No such file or directory".
cp ./summary.json "$FLYTE_OUTPUT_SUMMARY"
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
    # Declared outputs are what make this task consumable. `File` and `Dir` only, rejected
    # when the task is defined: a scalar would have to be parsed out of stdout, which is
    # silently wrong for any script that logs.
    outputs={"summary": File},
    # Who moves the bytes. "connector" (the default) hands the script a local path and the
    # connector uploads afterwards -- nothing needed on the node, but refused above 100 MB
    # (FLYTE_SLURM_CONNECTOR_UPLOAD_MAX_BYTES on the connector deployment). "job" hands it
    # the object-storage URI to upload to itself: one hop, no size limit, but the node then
    # needs a client and credentials. Choose it up front for a large artifact -- the
    # refusal only fires once the job has already done its work.
    output_upload="connector",
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
async def pipeline(epochs: int = 3, dataset_uri: str = "s3://my-bucket/datasets/demo") -> dict[str, str]:
    """Submit the script on Slurm, then read its output back on Kubernetes."""
    summary = await train(epochs=epochs, dataset_uri=dataset_uri)
    return await summarize(summary=summary)


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(pipeline, epochs=3)
    print("run url:", run.url)
    run.wait()
    print(run.outputs())
