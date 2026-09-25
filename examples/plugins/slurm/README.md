# Slurm plugin examples

| File | Shows |
|---|---|
| `slurm_script_example.py` | An existing sbatch script run unchanged. No container, no SDK in the image, no typed outputs. Currently the only way to run genuine multi-node work. |
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

`slurm_script` tasks need none of that: no image, no raw-data path, no credentials on
your machine.

## Cluster-side prerequisites

- The task image must be pullable by **Enroot on the compute nodes** -- a separate
  credential from your `docker login` and from Kubernetes `imagePullSecrets`. Either
  publish the image or write `~/.config/enroot/.credentials` for the submitting user.
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
