from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from flyte._task_plugins import TaskPluginRegistry
from flyte.connectors import AsyncConnectorExecutorMixin
from flyte.extend import AsyncFunctionTaskTemplate, TaskTemplate
from flyte.models import NativeInterface, SerializationContext

TASK_TYPE_NATIVE = "slurm"
TASK_TYPE_SCRIPT = "slurm_script"

_SBATCH_FIELDS = (
    "partition",
    "nodes",
    "ntasks",
    "cpus_per_task",
    "gres",
    "gpus_per_node",
    "mem",
    "time_limit",
    "account",
    "qos",
    "reservation",
    "constraint",
)


@dataclass
class Slurm:
    """Configuration for running a task on a Slurm cluster.

    Scheduling fields map one-to-one onto ``sbatch`` options; anything not covered
    goes in ``sbatch_options`` verbatim. Connection fields may be left unset and
    supplied cluster-wide on the connector via ``FLYTE_SLURM_HOST``,
    ``FLYTE_SLURM_USERNAME`` and ``FLYTE_SLURM_SSH_PRIVATE_KEY`` instead.

    Do not set ``resources`` on a Slurm task environment: the allocation is
    described here and granted by Slurm, not by Kubernetes.

    Attributes:
        partition: Slurm partition to submit to.
        nodes: Number of nodes to allocate.
        ntasks: Number of tasks (``--ntasks``). Leave unset for a single-process task.
        cpus_per_task: CPUs per task.
        gres: Generic resources, e.g. ``"gpu:8"``.
        gpus_per_node: GPUs per node, e.g. ``8`` or ``"h100:8"``.
        mem: Memory per node, e.g. ``"64G"``.
        time_limit: Wall-clock limit in Slurm format, e.g. ``"4:00:00"``.
        account: Account to charge.
        qos: Quality of service.
        reservation: Reservation name.
        constraint: Node feature constraint.
        sbatch_options: Extra ``--<key>=<value>`` options passed through verbatim.
            Use ``True`` for a bare flag. Overrides the first-class fields on conflict.
        container_image: Override the image submitted to Pyxis, e.g. a pre-imported
            squashfs path on the shared filesystem. Defaults to the task's image.
        container_mounts: ``--container-mounts`` entries, e.g. ``["/data:/data"]``.
        container_workdir: ``--container-workdir``.
        srun_args: Extra arguments inserted before the command on the ``srun`` line.
        env: Environment variables exported into the job, e.g. object-storage settings
            the Flyte entrypoint needs on the cluster.
        working_dir: Directory on the cluster for scripts and logs. Relative paths are
            under the SSH user's home. Defaults to ``.flyte/jobs``.
        host: Login node hostname.
        port: SSH port.
        username: SSH user jobs are submitted as.
        ssh_private_key: Name of the Flyte secret holding the SSH private key.
        known_hosts: Path to a known_hosts file on the connector for host-key verification.
        skip_host_key_verification: Disable host-key verification. Not for production.
    """

    partition: Optional[str] = None
    nodes: Optional[int] = None
    ntasks: Optional[int] = None
    cpus_per_task: Optional[int] = None
    gres: Optional[str] = None
    gpus_per_node: Optional[Any] = None
    mem: Optional[str] = None
    time_limit: Optional[str] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    reservation: Optional[str] = None
    constraint: Optional[str] = None
    sbatch_options: Dict[str, Any] = field(default_factory=dict)

    container_image: Optional[str] = None
    container_mounts: List[str] = field(default_factory=list)
    container_workdir: Optional[str] = None
    srun_args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    working_dir: Optional[str] = None

    host: Optional[str] = None
    port: int = 22
    username: Optional[str] = None
    ssh_private_key: Optional[str] = None
    known_hosts: Optional[str] = None
    skip_host_key_verification: bool = False

    def to_custom_config(self) -> Dict[str, Any]:
        sbatch = {name: getattr(self, name) for name in _SBATCH_FIELDS if getattr(self, name) is not None}
        connection: Dict[str, Any] = {}
        if self.port != 22:
            connection["port"] = self.port
        for name in ("host", "username", "known_hosts"):
            if getattr(self, name):
                connection[name] = getattr(self, name)
        if self.skip_host_key_verification:
            connection["skip_host_key_verification"] = True

        cfg: Dict[str, Any] = {
            "connection": connection,
            "sbatch": sbatch,
            "sbatch_options": dict(self.sbatch_options),
            "container": {
                "image": self.container_image,
                "mounts": list(self.container_mounts),
                "workdir": self.container_workdir,
                "srun_args": list(self.srun_args),
            },
            "env": dict(self.env),
        }
        if self.working_dir:
            cfg["working_dir"] = self.working_dir
        if self.ssh_private_key:
            cfg["secrets"] = {"ssh_private_key": self.ssh_private_key}
        return cfg


class SlurmFunctionTask(AsyncConnectorExecutorMixin, AsyncFunctionTaskTemplate):
    """A Python task executed as a Slurm job.

    The task's own container image and Flyte entrypoint are submitted through
    Pyxis/Enroot, so inputs, outputs, caching and retries behave exactly as they
    would for the same task running as a Kubernetes pod.
    """

    plugin_config: Slurm

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = TASK_TYPE_NATIVE

    def custom_config(self, sctx: SerializationContext) -> Dict[str, Any]:
        return self.plugin_config.to_custom_config()


class SlurmScriptTask(AsyncConnectorExecutorMixin, TaskTemplate):
    """An existing sbatch script run as a Flyte task, unmodified.

    Scalar inputs are exposed to the script as ``FLYTE_INPUT_<NAME>`` environment
    variables. The task reports phase, exit code and logs; it has no typed outputs.
    """

    def __init__(
        self,
        name: str,
        script: str,
        plugin_config: Slurm,
        inputs: Optional[Dict[str, Type]] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            interface=NativeInterface({k: (v, None) for k, v in inputs.items()} if inputs else {}, {}),
            task_type=TASK_TYPE_SCRIPT,
            image=None,
            **kwargs,
        )
        self.script = script
        self.plugin_config = plugin_config

    def custom_config(self, sctx: SerializationContext) -> Dict[str, Any]:
        cfg = self.plugin_config.to_custom_config()
        cfg["script"] = self.script
        return cfg


TaskPluginRegistry.register(Slurm, SlurmFunctionTask)
