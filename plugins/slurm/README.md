# Slurm Plugin for Flyte

Run Flyte 2 tasks on an existing Slurm cluster. Jobs are submitted over SSH to a
login node, so this works with any Slurm installation — including clusters managed
by [Soperator](https://github.com/nebius/soperator) — without changing the cluster.

## Installation

```bash
pip install flyteplugins-slurm
```

The connector must also be installed in the `flyteconnector` image of your Flyte
dataplane, and the `slurm` and `slurm_script` task types routed to the connector
service. See the deployment section below.

## Task types

| Task type | What is submitted | Typed I/O | Caching |
|---|---|---|---|
| `slurm` | The task's own container image and Flyte entrypoint, via Pyxis/Enroot | Yes | Yes |
| `slurm_script` | A user-supplied `sbatch` script, as-is | `File`/`Dir` outputs, declared | No |

> **`slurm` tasks need a container runtime on the cluster.** The native task type runs
> your image on the node, which requires either Pyxis/Enroot (the default) or Apptainer.
> Check which the cluster has before you start:
>
> ```bash
> scontrol show config | grep -i plugstack      # Pyxis: look for spank_pyxis.so
> command -v apptainer                          # the alternative
> ```
>
> Select it with `container_runtime`; see [Container runtimes](#container-runtimes). A
> cluster with neither cannot run native tasks -- use `slurm_script` and invoke whatever
> the site provides from the script.

## Python tasks on Slurm

```python
import flyte
from flyteplugins.slurm import Slurm

env = flyte.TaskEnvironment(
    name="train",
    plugin_config=Slurm(
        partition="main",
        nodes=1,
        gres="gpu:8",
        time_limit="4:00:00",
        # Where the Flyte entrypoint finds object storage from inside the job:
        env={"AWS_ENDPOINT_URL": "https://storage.example", "AWS_REGION": "eu-north1"},
        # Connection — or set FLYTE_SLURM_HOST / FLYTE_SLURM_USERNAME /
        # FLYTE_SLURM_SSH_PRIVATE_KEY on the connector and omit these:
        host="login.slurm.example.com",
        username="flyte",
        ssh_private_key="slurm-ssh-key",  # name of a Flyte secret
    ),
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-slurm"),
)

@env.task
async def train(steps: int) -> float:
    ...
```

Delete `plugin_config` and the same task runs as a Kubernetes pod. Nothing else changes.

The job runs `srun --container-image=<task image> a0 ...` inside an `sbatch` allocation
described by the `Slurm` fields. Anything `sbatch` accepts that is not a first-class
field goes in `sbatch_options`:

```python
Slurm(partition="main", sbatch_options={"exclusive": True, "mail-type": "FAIL"})
```

Do not set `resources` on a Slurm task environment. The allocation is described by the
`Slurm` config and granted by Slurm, not by Kubernetes.

### Container runtimes

`container_runtime` picks how the image is launched on the node. It defaults to `pyxis`,
which is what NVIDIA-shaped GPU clusters ship; `apptainer` covers the traditional HPC
sites where Pyxis is not installed. Nothing else about the job changes -- the directives,
exports and entrypoint are identical, so a task moves between clusters by changing this
one field.

```python
Slurm(partition="main", container_runtime="apptainer")
```

| | Pyxis | Apptainer |
|---|---|---|
| How it launches | flags on `srun` | a command the job runs |
| Image reference | `ghcr.io#org/img:tag` | `docker://ghcr.io/org/img:tag` |
| Mounts | `--container-mounts` | `--bind` |
| Working directory | `--container-workdir` | `--pwd` |
| Local image | `.sqsh` path | `.sif` path |

Both are given the same `container_mounts` and `container_workdir`; the plugin renders
whichever form the runtime wants, and rewrites the image reference accordingly.

**GPUs under Apptainer.** Apptainer does not expose the host's GPU driver and libraries
unless asked, so `--nv` is added automatically when the job requests GPUs through `gres`
or `gpus_per_node`. Without it the container starts, sees no device, and the failure
reads as a broken CUDA install rather than a missing flag. Enroot binds the NVIDIA stack
itself, so Pyxis needs no equivalent. On AMD, pass `container_args=["--rocm"]`; an
explicit GPU flag suppresses the inferred `--nv`.

**Tooling behind environment modules.** Many HPC sites keep `apptainer` off the default
PATH and expose it through Lmod or environment-modules, so the job cannot find it. List
what the job needs and the plugin emits the loads in the sbatch body, before `srun`:

```python
Slurm(
    partition="main",
    container_runtime="apptainer",
    modules=["apptainer", "cuda/12.2"],
    gres="gpu:8",
)
```

renders

```bash
module load apptainer
module load cuda/12.2
...
srun --nodes=1 --ntasks=1 apptainer exec --nv docker://<image> ... a0 ...
```

An unknown value is rejected where the task is defined rather than at submission, so a
typo surfaces to the task author instead of in the connector's logs.

### Container images

The runtime pulls the task image from its registry and caches it on the cluster. On
clusters where images are pre-imported to the shared filesystem, point at the local file
directly -- a `.sqsh` for Pyxis, a `.sif` for Apptainer:

```python
Slurm(container_image="/jail/images/train.sqsh", ...)
```

### Object storage from inside the job

The Flyte entrypoint inside the job reads inputs and writes outputs to the run's object
storage. Slurm worker nodes need network access to that storage and credentials for it.
Pass endpoint and credential *references* through `env`; keep secrets on the cluster
(for example via instance credentials or a credentials file on the shared filesystem),
not in the task config.

## Existing sbatch scripts

```python
from flyteplugins.slurm import Slurm, SlurmScriptTask

legacy_train = SlurmScriptTask(
    name="legacy_train",
    script=open("train.sbatch").read(),
    plugin_config=Slurm(partition="main", host=..., username=..., ssh_private_key=...),
    inputs={"epochs": int},
)
```

The script is submitted unchanged. Scalar inputs are exported as `FLYTE_INPUT_<NAME>`
environment variables, and `File`/`Dir` inputs as their URI.

### Outputs from a script task

Declare them, and a downstream task can consume the script's results:

```python
train = SlurmScriptTask(
    name="train",
    script=open("train.sbatch").read(),
    plugin_config=Slurm(partition="main"),
    inputs={"epochs": int},
    outputs={"model": File, "shards": Dir},
)
```

A script cannot write Flyte's own output format, so the plugin hands it a destination URI
per output and the script writes there with whatever tooling the site already uses:

```bash
#SBATCH ...
python train.py --epochs "$FLYTE_INPUT_EPOCHS" --out ./model.pt
aws s3 cp ./model.pt "$FLYTE_OUTPUT_MODEL"          # or gcloud storage cp, rclone, ...
```

The bytes go straight from the job to object storage — the job already holds credentials
for reading its inputs — so nothing large passes through the connector. Once the job
succeeds, the connector checks each destination exists and records it as the declared
`File` or `Dir`.

Two constraints:

- **`File` and `Dir` only.** A scalar output would mean parsing stdout, which is silently
  wrong for any script that logs.
- **A declared output the script never wrote fails the task**, even on exit 0. The
  alternative is handing a downstream task a URI to nothing, which surfaces much later as
  an unexplained read error.

The script's own leading `#SBATCH` directives are hoisted above the generated `export`
lines and the plugin's directives follow them, so non-conflicting options are kept and
the plugin's win on a duplicate — `sbatch` applies options in order and takes the last.
Both blocks must sit above any executable line, because `sbatch` stops reading directives
there; a leading shebang in the script is dropped.

## How states map

| Slurm | Flyte |
|---|---|
| `PENDING`, `CONFIGURING`, `REQUEUED`, `SUSPENDED` | `QUEUED` — does not count as running |
| `RUNNING`, `COMPLETING` | `RUNNING` |
| `COMPLETED` | `SUCCEEDED` |
| `FAILED`, `NODE_FAIL`, `OUT_OF_MEMORY`, `TIMEOUT`, `DEADLINE`, `BOOT_FAIL` | `FAILED` — with Slurm's reason and the tail of stderr |
| `PREEMPTED` | `RETRYABLE_FAILED` |
| `CANCELLED` | `ABORTED` |

Aborting the Flyte run runs `scancel`.

## Deployment

1. **Connector image.** Add `flyteplugins-slurm` to the `flyteconnector` image
   (`maint_tools/build_default_image.py` in this repo builds the default one).
2. **Routing.** In the dataplane values:
   ```yaml
   union:
     configmap:
       enabled_plugins:
         tasks:
           task-plugins:
             default-for-task-types:
               slurm: connector-service
               slurm_script: connector-service
   ```
3. **Credentials.** The SSH private key is a Flyte secret named in
   `Slurm.ssh_private_key`, or set cluster-wide as `FLYTE_SLURM_SSH_PRIVATE_KEY` on
   the `flyteconnector` deployment. Provide a `known_hosts` file for host-key
   verification; `skip_host_key_verification=True` exists for development only.
4. **Network.** The connector needs to reach the login node on its SSH port. Restrict
   the login node's source ranges to the connector's egress addresses.

The connector keeps one SSH connection per cluster, reused across calls and
re-established if it drops, so tracking many jobs costs one login-node session rather
than one per job. It does still issue one `squeue` per job per poll: `get` is called per
resource, so coalescing would need a cache inside the connector. The transport already
accepts several ids per call, which is where that would plug in.

Each job leaves a `.sbatch`, `.out` and `.err` file in `working_dir` and nothing removes
them, by design -- they are the first thing to read when a job fails. On a busy cluster
they accumulate in the submitting user's home, so prune them on whatever schedule suits
the site.

## Connector-level defaults

| Variable | Purpose |
|---|---|
| `FLYTE_SLURM_HOST` | Login node when the task config leaves `host` unset |
| `FLYTE_SLURM_PORT` | SSH port (default 22) |
| `FLYTE_SLURM_USERNAME` | SSH user |
| `FLYTE_SLURM_SSH_PRIVATE_KEY` | Private key contents |
| `FLYTE_SLURM_KNOWN_HOSTS` | Path to a known_hosts file on the connector |
| `FLYTE_SLURM_WORKING_DIR` | Directory for scripts and logs (default `.flyte/jobs` under the user's home) |

## Retries and preemption

`PREEMPTED` maps to `RETRYABLE_FAILED`, which only re-submits when the task asks for it:
set `retries` on the task, since the default is 0 and a preempted job otherwise ends the
run. Checkpoint to the cluster's shared filesystem if the work is long, so a retry
resumes rather than starting over.

## Known gaps

What the plugin does not do today, and what to do instead.

**Execution**

- **No multi-node gang execution for `slurm` tasks.** The native task pins
  `srun --nodes=1 --ntasks=1`, so the entrypoint runs exactly once even when the
  allocation spans several nodes. Without the pin, `nodes=2` starts one entrypoint per
  node and each writes the same output prefix. Distributed work belongs in a
  `slurm_script` task, which drives `srun` or `mpirun` itself.
- **Only Pyxis and Apptainer are supported as container runtimes.** Anything else needs a
  new branch in `_container_invocation`. A cluster with neither cannot run native tasks;
  use `slurm_script` there.
- **`resources` is refused on a Slurm task environment**, rather than silently ignored.
  Use `cpus_per_task`, `mem`, `gres` or `gpus_per_node`.

**Data and I/O**

- **`slurm_script` outputs are `File`/`Dir` only**, and only when declared. A scalar
  would have to come out of stdout. Caching is still a no-op for script tasks.
- **Script inputs are limited to scalars and URIs.** `str`, `int`, `float` and `bool`
  become `FLYTE_INPUT_<NAME>`; `File` and `Dir` become their URI. Anything else fails at
  submission.
- **No clickable log links.** Job output lives on the login node, not behind a URL, so
  the paths are named in the task's message instead; live stdout is streamed through the
  connector.

**Operations**

- **SSH transport only.** A `slurmrestd` transport fits behind
  `flyteplugins.slurm.transport.SlurmTransport` but is not implemented. Clusters often
  have the prerequisite (`AuthAltTypes=auth/jwt`) without running the daemon.
- **One identity.** Every job runs as the configured SSH user, so the cluster attributes
  all work to that account regardless of who launched the run.
- **Status is polled per job.** One SSH connection per cluster is reused, but `get` is
  called per resource, so it issues one `squeue` per job per poll. Coalescing would need
  a cache in the connector.
- **Job files accumulate.** Nothing removes the `.sbatch`, `.out` and `.err` left in
  `working_dir`; they are the first thing to read when a job fails, so prune them on
  whatever schedule suits the site.
- **A cluster without accounting has a small blind spot.** With `sacct` unavailable a
  finished job is resolved through `scontrol`, which keeps it only for `MinJobAge`
  seconds; one that finishes and ages out between polls cannot be resolved.

**Security**

- **Values in `env` are written to the cluster in plain text**, inside the generated
  sbatch script on the login node's filesystem. Mount credentials from the shared
  filesystem and reference the path instead.
- **The Apptainer path is unit-tested only.** It has not been exercised against a real
  Apptainer cluster.
