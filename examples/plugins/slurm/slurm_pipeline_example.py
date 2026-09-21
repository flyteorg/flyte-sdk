"""A three-step pipeline that runs one step on Slurm and the rest on Kubernetes.

    prepare  (Kubernetes pod)  -- builds a dataset, returns Dir
       |
    train    (Slurm job)       -- reads the Dir, returns a model File
       |
    evaluate (Kubernetes pod)  -- reads the File, returns metrics

Only `train` carries a `plugin_config`, and that is the only difference between the
environments. The handoff in both directions is the run's object storage: `Dir` and
`File` contents are offloaded there, so the literal passed between tasks is just a URI.
Nothing moves through the control plane, and neither side needs to know where the other
ran.

Two rules this example exists to demonstrate:

  * Return `File`/`Dir`, never a filesystem path. `"/data/model.pt"` would satisfy the
    type checker and then fail at `open()` in the pod -- the Nebius filesystem does not
    exist on GKE. Path references are valid only between tasks that share a filesystem.
  * Keep the bytes out of the control plane. Large payloads belong in `File`/`Dir`;
    scalars and small dicts can be returned directly.

    flyte run slurm_pipeline_example.py pipeline
"""

import os
import pathlib

from flyteplugins.slurm import Slurm

import flyte
from flyte.io import Dir, File

image = flyte.Image.from_debian_base()

# Same image, same code path as the pod environment below -- only the placement differs.
slurm_env = flyte.TaskEnvironment(
    name="slurm-pipeline-train",
    image=image,
    plugin_config=Slurm(
        partition="main",
        nodes=1,
        cpus_per_task=4,
        mem="8G",
        time_limit="2:00:00",
        # Add `gres="gpu:1"` only if the cluster declares GRES -- see slurm_example.py.
        container_mounts=["/home/flyte/.gcp:/etc/gcp:ro"],
        env={"GOOGLE_APPLICATION_CREDENTIALS": "/etc/gcp/sa.json"},
    ),
)

# Runs as an ordinary Kubernetes pod: no plugin_config.
#
# `depends_on` lists the environments to deploy alongside this one, so it points from the
# environment holding the task you invoke to the environments its tasks call into. Only
# the invoked environment and this closure are built, so declaring it the other way round
# builds cleanly and then fails at run time with:
#
#     Environment 'slurm-pipeline-train' not found in image cache.
k8s_env = flyte.TaskEnvironment(
    name="slurm-pipeline-k8s",
    image=image,
    resources=flyte.Resources(cpu="1", memory="1Gi"),
    depends_on=[slurm_env],
)


@k8s_env.task
async def prepare(rows: int = 1_000) -> Dir:
    """Upstream step on Kubernetes: produce a dataset in object storage."""
    local = pathlib.Path("/tmp/dataset")
    local.mkdir(parents=True, exist_ok=True)
    (local / "train.csv").write_text("\n".join(f"{i},{i * 2}" for i in range(rows)) + "\n")
    (local / "MANIFEST").write_text(f"rows={rows}\n")
    return await Dir.from_local(local)


# `cache`/`retries` are per-task. Caching means an unchanged re-run skips the Slurm job
# entirely; RETRYABLE_FAILED on PREEMPTED means preemption costs a retry, not the run.
@slurm_env.task(cache="auto", retries=2)
async def train(dataset: Dir, steps: int = 100) -> File:
    """The Slurm step: downloads the dataset, trains, uploads the model.

    `dataset.download()` pulls from object storage into the job's scratch space on the
    worker. For data read once per run that is the right thing. For a training set read
    every epoch, stage it onto the cluster's shared filesystem instead and pass a path --
    re-reading across clouds on every epoch is the expensive mistake.
    """
    local_dir = pathlib.Path(await dataset.download())
    rows = sum(1 for _ in (local_dir / "train.csv").open())

    model = pathlib.Path("/tmp/model.txt")
    model.write_text(
        f"rows={rows} steps={steps} node={os.uname().nodename} slurm_job={os.environ.get('SLURM_JOB_ID', 'unset')}\n"
    )
    return await File.from_local(model)


@k8s_env.task
async def evaluate(model: File) -> dict[str, str]:
    """Downstream step back on Kubernetes: read what the Slurm job produced."""
    async with model.open("rb") as fh:
        summary = (await fh.read()).decode().strip()
    return {"model_summary": summary, "evaluated_on": os.uname().nodename}


@k8s_env.task
async def pipeline(rows: int = 1_000, steps: int = 100) -> dict[str, str]:
    """Driver: sequences the three steps. Runs as a pod; only `train` lands on Slurm."""
    dataset = await prepare(rows=rows)
    model = await train(dataset=dataset, steps=steps)
    return await evaluate(model=model)


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(pipeline, rows=1_000, steps=100)
    print("run url:", run.url)
    run.wait()
    print(run.outputs())
