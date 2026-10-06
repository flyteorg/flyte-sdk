# Slurm plugin examples

| File | Shows |
|---|---|
| `slurm_script_example.py` | An existing sbatch script run unchanged — no container, no SDK in the image — declaring a `File` output that a Kubernetes task then reads. Also the only way to run genuine multi-node work. |
| `slurm_example.py` | A Python task on Slurm with the same typed I/O it would have on Kubernetes. Delete `plugin_config` and it runs as a pod. |
| `slurm_pipeline_example.py` | Three steps, two backends: prepare on Kubernetes, train on Slurm, evaluate on Kubernetes, with a `Dir` in and a `File` out. |

## Running them

Registered, against a dataplane whose connector has the plugin installed and
`FLYTE_SLURM_*` configured:

```bash
flyte run slurm_pipeline_example.py pipeline
```

Locally, driving the connector in-process. A native `slurm` task needs a **remote**
raw-data path -- with a local one the run never reaches the cluster -- and the plugin
wheel has to be baked into the task image:

```bash
export FLYTE_SLURM_HOST=... FLYTE_SLURM_USERNAME=... \
       FLYTE_SLURM_SSH_PRIVATE_KEY="$(cat key)" FLYTE_SLURM_KNOWN_HOSTS=./known_hosts
export _F_LOCAL_PLUGINS=flyteplugins-slurm        # while the plugin is unreleased
export GOOGLE_APPLICATION_CREDENTIALS=~/.gcp/sa.json   # the *laptop* uploads too

flyte run --local --raw-data-path gs://<bucket>/scratch slurm_example.py train
```

`slurm_script` tasks need none of that on your machine: no image, no raw-data path, no
credentials. With the default `output_upload="connector"` the compute node needs none
either -- the connector moves the bytes.

## Outputs from a script task

A script cannot write Flyte's own output format, so it is handed a destination per declared
output and writes there. `File` and `Dir` only, rejected when the task is defined: a scalar
would have to be parsed out of stdout, which is silently wrong for any script that logs.

```python
outputs={"summary": File}        # exported to the script as FLYTE_OUTPUT_SUMMARY
```

`output_upload` decides what that destination is and who moves the bytes:

| | `"connector"` (default) | `"job"` |
|---|---|---|
| `FLYTE_OUTPUT_SUMMARY` holds | a local path | the object-storage URI |
| The script writes it with | `cp ./summary.json "$FLYTE_OUTPUT_SUMMARY"` | `aws s3 cp`, `rclone copyto`, `gcloud storage cp`, ... |
| The compute node needs | nothing | a client and credentials for the store |
| Size limit | 100 MB, then the task fails | none |

The default asks nothing of the cluster, which suits most script output — summaries,
metrics, small models. Above the ceiling the task fails telling you to switch to `"job"`,
and since the mode decides what the script was handed, that cannot be fixed after the
fact: the job's work is lost. Choose `"job"` up front for anything large. Operators can
raise the ceiling with `FLYTE_SLURM_CONNECTOR_UPLOAD_MAX_BYTES` on the connector
deployment (`0` disables it).

Either way, once the job succeeds the connector checks each destination exists — a declared
output the script never wrote fails the task, even on exit 0 — and records it, so a
downstream task consumes it as an ordinary `File`.

## Cluster-side prerequisites

- The task image must be pullable by **Enroot on the compute nodes** -- a separate
  credential from your `docker login` and from Kubernetes `imagePullSecrets`. Either
  publish the image or write `~/.config/enroot/.credentials` for the submitting user.
  On an Apptainer cluster, set `container_runtime="apptainer"` instead (plus
  `container_args=["--nv"]` for GPUs, and `modules=["apptainer"]` if it is an env module).
- The job reads inputs and writes outputs from inside the container, so the worker needs
  credentials for the run's object storage. Mount them from the shared filesystem and
  reference the path in `env`; never put a secret in `env` itself, which is rendered into
  the sbatch script in plain text.
- Do not set `resources` on a Slurm task environment. The allocation is described by the
  `Slurm` fields and granted by Slurm.

## Reading a failure

Every job leaves its script and logs on the login node under `~/.flyte/jobs`:

```bash
ls -t ~/.flyte/jobs | head
cat  ~/.flyte/jobs/<job>.sbatch    # exactly what was submitted; re-runnable by hand
tail ~/.flyte/jobs/<job>.err
```

Reading the generated `.sbatch` answers most questions outright, and running it with
`sbatch` by hand separates a plugin bug from a cluster problem.
