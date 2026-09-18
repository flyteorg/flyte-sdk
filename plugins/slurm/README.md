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
| `slurm_script` | A user-supplied `sbatch` script, as-is | No — phase, exit code and logs | No |

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

### Container images

Pyxis pulls the task image from its registry and caches it on the cluster. On clusters
where images are pre-imported to the shared filesystem, point at the squashfs directly:

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
environment variables. Our `#SBATCH` directives come first, so the script's own
directives are kept but overridden where they conflict.

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

The connector keeps one SSH connection per cluster and batches status queries, so it
opens one login-node session per poll, not one per job.

## Connector-level defaults

| Variable | Purpose |
|---|---|
| `FLYTE_SLURM_HOST` | Login node when the task config leaves `host` unset |
| `FLYTE_SLURM_PORT` | SSH port (default 22) |
| `FLYTE_SLURM_USERNAME` | SSH user |
| `FLYTE_SLURM_SSH_PRIVATE_KEY` | Private key contents |
| `FLYTE_SLURM_KNOWN_HOSTS` | Path to a known_hosts file on the connector |
| `FLYTE_SLURM_WORKING_DIR` | Directory for scripts and logs (default `.flyte/jobs` under the user's home) |

## Not yet supported

- Multi-node gang jobs with a distributed launcher (`nodes > 1` allocates, but the
  entrypoint runs as a single process).
- A slurmrestd transport. The SSH transport is behind a small protocol
  (`flyteplugins.slurm.transport.SlurmTransport`) so one can be added without touching
  the connector.
- Per-user job attribution; jobs run as the configured SSH user.
