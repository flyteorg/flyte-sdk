"""Run a Python task on Slurm with the same typed I/O it would have on Kubernetes.

A native `slurm` task submits the task's *own* container image and the Flyte entrypoint
as an sbatch job under Pyxis/Enroot. Inputs, outputs, caching and retries behave exactly
as they do for a pod -- delete `plugin_config` and this runs on Kubernetes unchanged,
which also makes it the quickest way to tell a Slurm problem from a task problem.

    # locally, the raw-data path must be remote or the run never reaches Slurm
    export _F_LOCAL_PLUGINS=flyteplugins-slurm
    flyte run --local --raw-data-path gs://<bucket>/scratch slurm_example.py train

    # registered: connection details and the raw-data path come from the dataplane
    flyte run slurm_example.py train
"""

import os
import pathlib

from flyteplugins.slurm import Slurm

import flyte
from flyte.io import File

image = flyte.Image.from_debian_base()

env = flyte.TaskEnvironment(
    name="slurm-native",
    image=image,
    plugin_config=Slurm(
        partition="main",
        nodes=1,
        cpus_per_task=4,
        mem="8G",
        time_limit="1:00:00",
        # To request GPUs, add `gres="gpu:1"` or `gpus_per_node=1` -- but only if the
        # cluster actually declares GRES. On a cluster without it, sbatch rejects the job
        # outright with `Invalid generic resource (gres) specification`. Check with
        # `sinfo -N -o "%N %G"` before adding it.
        # Credentials for the run's object storage, mounted from the cluster's shared
        # filesystem. Never put secrets in `env` -- it is rendered into the sbatch script
        # in plain text.
        container_mounts=["/home/flyte/.gcp:/etc/gcp:ro"],
        env={"GOOGLE_APPLICATION_CREDENTIALS": "/etc/gcp/sa.json"},
        # Anything sbatch accepts that is not a first-class field. (`exclusive` also
        # works here, but it reserves the whole node -- don't copy that onto a shared
        # cluster without meaning it.)
        sbatch_options={"requeue": True},
    ),
    # Do NOT set `resources` here: the allocation comes from the Slurm fields above and
    # is granted by Slurm, not Kubernetes.
)


# `cache` and `retries` are per-task, not per-environment. PREEMPTED maps to
# RETRYABLE_FAILED, so a preempted allocation is retried rather than failing the run.
@env.task(cache="auto", retries=2)
async def train(steps: int = 100) -> File:
    """Write a 'model' to the run's object storage and return a reference to it."""
    out = pathlib.Path("/tmp/model.txt")
    out.write_text(f"trained for {steps} steps on {os.uname().nodename}\n")
    # File.from_local uploads to the run's raw-data path, so any task -- on Slurm or on
    # Kubernetes -- can read it afterwards.
    return await File.from_local(out)


@env.task
async def where_am_i() -> str:
    """Smallest possible proof the task body really ran on a Slurm worker.

    A laptop hostname here means the run short-circuited locally and never reached the
    cluster -- usually a missing remote `--raw-data-path`.
    """
    return f"{os.uname().nodename} (SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', 'unset')})"


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(train, steps=500)
    print("run url:", run.url)
    run.wait()
