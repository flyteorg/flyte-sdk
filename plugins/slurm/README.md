# Slurm Plugin for Flyte

Run Flyte 2 tasks on an existing Slurm cluster. Jobs are submitted over SSH to a
login node, so this works with any Slurm installation — including clusters managed
by [Soperator](https://github.com/nebius/soperator) — without changing the cluster. It assumes
nothing about the cloud the cluster runs in, or about where the run's object storage lives.

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
| `slurm_script` | A user-supplied `sbatch` script, as-is | `File`/`Dir` outputs, declared | Yes |

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

### Directives in your script

The script's own leading `#SBATCH` directives are hoisted above the generated `export`
lines and the plugin's directives follow them, so non-conflicting options are kept and
the plugin's win on a duplicate — `sbatch` applies options in order and takes the last.
Both blocks must sit above any executable line, because `sbatch` stops reading directives
there; a leading shebang in the script is dropped.

### Outputs from a script task

An arbitrary sbatch script cannot write Flyte's literal format, so a script task produces
only what it is told to produce. Declaring outputs is what lets a downstream task consume
the results; without them the task returns nothing.

#### Declare what the script will write

```python
train = SlurmScriptTask(
    name="train",
    script=open("train.sbatch").read(),
    plugin_config=Slurm(partition="main"),
    inputs={"epochs": int},
    outputs={"model": File, "shards": Dir},
)
```

**`File` and `Dir` only**, rejected at definition time rather than at run time. A scalar
would have to be parsed out of stdout, which is silently wrong for any script that logs, and
a structured value has no representation a shell script can write. (A native `slurm` task
runs Flyte's entrypoint and so has the full range of output types.)

**The two sides must match:** every `$FLYTE_OUTPUT_*` the script writes to has to be declared
in `outputs`, and every declared output has to be written. Both are checked, each as early as
it can be:

| The script has | `outputs` has | What happens |
|---|---|---|
| `$FLYTE_OUTPUT_MODEL` | `{"model": File}` | Recorded, and consumable downstream |
| `$FLYTE_OUTPUT_MODEL` | nothing, or another name | `ValueError` when the task is defined |
| nothing | `{"model": File}` | The task fails when the job finishes, even on exit 0 |

The variable is only exported for a declared output, so without the first check the job fails
on the cluster with `FLYTE_OUTPUT_MODEL: unbound variable` -- or, in a script without
`set -u`, writes to the empty path and can still exit 0 having produced nothing. Only `$NAME`
and `${NAME}` expansions count, so a mention in a comment is not a reference and a name
assembled at run time is left alone. The other direction has to wait for the job: writing
nothing would hand a downstream task a URI to nothing, which surfaces much later as an
unexplained read error.

#### Write to the destination the script is given

Each output arrives as `FLYTE_OUTPUT_<NAME>` — upper-cased, with anything not alphanumeric
replaced by `_`. By default it is an ordinary local path, so writing an output is a `cp`:

```bash
python train.py --epochs "$FLYTE_INPUT_EPOCHS" --out ./model.pt
cp ./model.pt "$FLYTE_OUTPUT_MODEL"
```

When the job succeeds the connector confirms each destination exists and records it as the
declared `File` or `Dir`.

#### Choose who uploads

The default has the connector move the bytes, which is why the script above needed no
credentials and no upload tool. `output_upload` switches it:

| | `"connector"` (default) | `"job"` |
|---|---|---|
| `FLYTE_OUTPUT_<NAME>` holds | a local path | the object-storage URI |
| Who uploads | the connector, after the job | the script, during the job |
| The compute node needs | nothing | a client and credentials for the store |
| The bytes travel | node → connector → storage | node → storage |
| Size limit | 100 MB by default | none |

**`output_upload="connector"`.** The script writes an ordinary file and the connector streams
it to object storage over the SSH connection it already holds. Nothing is asked of the
compute node — no upload tool, no credentials, no endpoint configuration. This suits what
most scripts emit: metrics, summaries, configs, small models.

It refuses above 100 MB. Streaming would work, but every byte would take two hops instead of
one, through a pod that is concurrently polling every other job this connector tracks, on
its bandwidth rather than the cluster's. Since the decision has to be made before the job
runs — it determines whether the script gets a path or a URI — the connector fails rather
than quietly taking the slow path:

```
Output 'model' of Slurm job 95 is 512 MB, above the 100 MB the connector will move on a
job's behalf. Set output_upload='job' on the task and have the script upload to the URI it
is given in FLYTE_OUTPUT_<NAME> ...
```

The job's work is lost when that happens, which is the cost of finding out at the end. If an
output might be large, choose `"job"` up front.

The ceiling is `FLYTE_SLURM_CONNECTOR_UPLOAD_MAX_BYTES` on the connector deployment — a plain
byte count or a suffixed size (`500MB`, `2GB`, `512MiB`), or `0` for no ceiling at all. It
lives there rather than on the task because it is the connector pod's bandwidth and scratch
space being spent, shared with every job it polls: a task that could raise it unilaterally
would be spending someone else's headroom. Whoever sized that pod can weigh a larger number
against how many jobs run at once. A value that is not a size fails the task that reads it
rather than falling back to the default, so a typo cannot quietly reinstate 100 MB.

**`output_upload="job"`.** The destination is the object-storage URI and the script uploads
directly, one hop, using the cluster's bandwidth. There is no size limit. The node needs a
client and credentials for the store — mount the credentials from the shared filesystem the
same way a native task does:

```python
train = SlurmScriptTask(..., outputs={"model": File}, output_upload="job")
```

```bash
# S3, and S3-compatible stores (MinIO, R2, Nebius, Ceph) with --endpoint-url
aws s3 cp ./model.pt "$FLYTE_OUTPUT_MODEL"

# Anything rclone has a remote for, which is usually already configured on HPC clusters
rclone copyto ./model.pt "$FLYTE_OUTPUT_MODEL"

# Google Cloud Storage
gcloud storage cp ./model.pt "$FLYTE_OUTPUT_MODEL"

# Azure Blob
azcopy copy ./model.pt "$FLYTE_OUTPUT_MODEL"

# A directory output: copy the tree, not a single file
aws s3 cp --recursive ./checkpoints "$FLYTE_OUTPUT_CHECKPOINTS"
```

Check what the node actually has before committing to one — `command -v aws rclone gcloud
azcopy` on a login node answers it. A script task runs on the bare node, not in a container,
so the tooling is the site's rather than your image's.

None of this applies to a native `slurm` task: its entrypoint writes outputs to object
storage itself, so there is no mode to choose and no size limit.

### Caching a script task

`cache="auto"` works, and a hit restores the declared outputs without submitting the job --
the whole point on a cluster where a miss can mean hours in a queue.

The version cannot come from a function, because there isn't one: the default policy would
return `sha256("")`, one constant shared by every script task, so an edited script would keep
hitting its old entry and two unrelated tasks would collide. The plugin computes it instead,
over what determines the result:

| Change | Cache |
|---|---|
| Script body | Invalidated |
| Declared output added, removed, or retyped | Invalidated |
| `partition`, `nodes`, `time_limit`, `gres`, `sbatch_options`, ... | Invalidated |
| `host`, `port`, `username`, `ssh_private_key`, `known_hosts` | Reused |
| `output_upload` | Reused |

The reused rows are deliberate. Moving to a new login node, rotating the SSH secret, or
changing who uploads the bytes does not change what the job computes -- and the output lands
at the same URI either way -- so discarding good results over it would be wrong.

`Cache(behavior="override", version_override=...)` is left untouched; the substitution only
happens for `"auto"`. With no outputs declared a hit still skips the job but restores
nothing, which is rarely useful -- declare outputs, or `cache="disable"`.

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

### Two ways to configure the connection

**In the task config, with Flyte secrets.** Nothing goes in the data plane's Helm values
beyond the connector image:

```python
Slurm(
    partition="main",
    host="login.example.com",
    username="flyte",
    ssh_private_key="slurm-ssh-key",        # name of a Flyte secret
    known_hosts_secret="slurm-known-hosts",  # name of a Flyte secret
)
```

Both secrets are named, not inlined. The platform resolves them and hands the values to
the connector, so no key or host entry appears in a task definition, an image, or a Helm
chart. `known_hosts_secret` carries the entries themselves rather than a path, which is
what removes the last reason to mount anything.

**On the connector, for a shared cluster.** When one Slurm cluster serves every task, a
platform team can set it once with `FLYTE_SLURM_HOST`, `FLYTE_SLURM_USERNAME`,
`FLYTE_SLURM_SSH_PRIVATE_KEY` and `FLYTE_SLURM_KNOWN_HOSTS` on the `flyteconnector`
deployment, and tasks carry only scheduling options. The connector's environment wins over
task config, so a task cannot redirect the deployment's shared key at a host of its
choosing.

The two compose: a connector with no environment set leaves everything to the task config,
which is also how local execution works.

`skip_host_key_verification=True` exists for development only, is not task-settable, and
must be enabled with `FLYTE_SLURM_SKIP_HOST_KEY_VERIFICATION` on the connector.


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
| `FLYTE_SLURM_CONNECTOR_UPLOAD_MAX_BYTES` | Largest script-task output the connector will move itself (default `100MB`; `0` for no limit) |

### The connector needs object-storage write access for `output_upload="connector"`

That default makes the connector pod a writer to the run's output prefix, which a connector
otherwise never is -- so the `flyteconnector` service account does not get the data plane's
cloud identity the way `union-system`, `webhook` and `dataproxy` do. Without it the pod
authenticates as the node's default identity and the upload fails *after* the job has
succeeded, with `403 ... storage.objects.create` (GCP) or `AccessDenied` (S3).

```yaml
flyteconnector:
  serviceAccount:
    annotations:
      iam.gke.io/gcp-service-account: union-system@<project>.iam.gserviceaccount.com
      # eks.amazonaws.com/role-arn: arn:aws:iam::<account>:role/<backend-role>
```

On GKE the annotation also needs the workload-identity binding
(`roles/iam.workloadIdentityUser` for `<project>.svc.id.goog[<namespace>/flyteconnector]`),
then a restart. Native tasks and `output_upload="job"` need none of this: the job writes with
the credentials it already holds for its inputs.

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
  would have to come out of stdout.
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
