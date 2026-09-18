"""Slurm connector plugin for Flyte.

Run Flyte 2 tasks on an existing Slurm cluster. The connector submits jobs over
SSH to a login node, so it works with any Slurm installation, including
Soperator-managed clusters.

Two task types are provided:

- ``slurm``: a Python task run inside its own container image via Pyxis/Enroot.
  Typed inputs and outputs, caching, retries and error reporting work as they
  do for a Kubernetes task. Remove ``plugin_config`` and the same task runs on
  Kubernetes.
- ``slurm_script``: an existing ``sbatch`` script submitted as-is. Phase and
  logs only.

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
        host="login.slurm.example.com",
        username="flyte",
        ssh_private_key="slurm-ssh-key",
    ),
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-slurm"),
)

@env.task
async def train(steps: int) -> float:
    ...
```
"""

from flyteplugins.slurm.connector import SlurmConnector, SlurmJobMetadata, SlurmScriptConnector
from flyteplugins.slurm.task import Slurm, SlurmFunctionTask, SlurmScriptTask

__all__ = [
    "Slurm",
    "SlurmConnector",
    "SlurmFunctionTask",
    "SlurmJobMetadata",
    "SlurmScriptConnector",
    "SlurmScriptTask",
]
