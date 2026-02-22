import os
import signal
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Union

import flyte
import flyte.report
from cloudpickle import cloudpickle
from flyte._context import internal_ctx
from flyte._logging import logger
from flyte._task import P, R
from flyte.extend import AsyncFunctionTaskTemplate, TaskPluginRegistry
from flyte.models import SerializationContext, TaskContext
from flyteidl2.plugins.kubeflow import common_pb2
from flyteidl2.plugins.kubeflow.pytorch_pb2 import (
    DistributedPyTorchTrainingReplicaSpec,
    DistributedPyTorchTrainingTask,
    ElasticConfig,
)
from google.protobuf.json_format import MessageToDict
from torch.distributed import run
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
from torch.distributed.launcher.api import LaunchConfig, elastic_launch


@dataclass
class RunPolicy:
    """
    RunPolicy describes some policy to apply to the execution of a kubeflow job.

    Args:
        clean_pod_policy (str, optional): Policy for cleaning up pods after the PyTorchJob completes.
            Allowed values are "None", "all", or "Running". Defaults to None.
        ttl_seconds_after_finished (int, optional): Defines the TTL (in seconds) for cleaning
            up finished PyTorchJobs. Defaults to None.
        active_deadline_seconds (int, optional): Specifies the duration (in seconds) since
            startTime during which the job can remain active before it is terminated.
            Must be a positive integer. Applies only to pods where restartPolicy is
            OnFailure or Always. Defaults to None.
        backoff_limit (int, optional): Number of retries before marking this job as failed.
            Defaults to None.
    """

    clean_pod_policy: Optional[Literal["None", "all", "Running"]] = None
    ttl_seconds_after_finished: Optional[int] = None
    active_deadline_seconds: Optional[int] = None
    backoff_limit: Optional[int] = None


@dataclass
class Elastic:
    """
    Elastic defines the configuration for running a PyTorch elastic job using torch.distributed.

    When a worker fails (e.g. CUDA OOM), the elastic agent detects the failure and
    restarts all workers as a group. Each restart cycle has a cost determined by the
    NCCL timeout settings below. The total worst-case time before the job fails is::

        (max_restarts + 1) * (nccl_collective_timeout_sec + nccl_heartbeat_timeout_sec)

    For example, with defaults (max_restarts=3, collective=600s, heartbeat=300s):
    4 * 900s = 60 min. With aggressive settings (max_restarts=0, collective=60s,
    heartbeat=60s): 1 * 120s = 2 min.

    Args:
        nnodes (Union[int, str]): Number of nodes to use. Can be a fixed int or a range
            string (e.g., "2:4" for elastic training).
        nproc_per_node (int): Number of processes to launch per node.
        rdzv_backend (literal): Rendezvous backend to use. Typically "c10d". Defaults to "c10d".
        run_policy (RunPolicy, optional): Run policy applied to the job execution.
            Defaults to None.
        monitor_interval (int): Interval (in seconds) the elastic agent polls worker
            process health. Once a worker process exits, detection takes at most this
            long. Defaults to 3.
        max_restarts (int): Maximum number of worker group restarts before the elastic
            agent gives up and raises ``ChildFailedError``. Each restart kills all
            workers and relaunches the entire group. If the failure is deterministic
            (e.g. model too large for GPU memory), restarts just repeat the same
            failure — set to 0 to fail immediately. Use higher values for transient
            failures (e.g. spot instance preemption, occasional OOM from variable
            batch sizes). Defaults to 3.
        rdzv_configs (Dict[str, Any]): Rendezvous configuration key-value pairs.
            Defaults to {"timeout": 900, "join_timeout": 900}.
        nccl_heartbeat_timeout_sec (Optional[int]): Timeout in seconds for the NCCL
            heartbeat monitor thread. After the collective timeout fires and the NCCL
            watchdog aborts the communicator, the heartbeat monitor waits this long
            before sending SIGABRT to kill the worker process. This is the second
            phase of failure detection — it converts a stuck NCCL abort into a hard
            process kill. Defaults to 300 (5 min) instead of PyTorch's 1800s (30 min).
            Set to None to use PyTorch default.
        nccl_async_error_handling (bool): When True, sets TORCH_NCCL_ASYNC_ERROR_HANDLING=1
            so that NCCL aborts stuck collectives asynchronously instead of blocking
            indefinitely. This causes the worker process to crash-exit on a stuck
            collective, which the elastic agent detects within ``monitor_interval``
            seconds (~3s by default) — much faster than waiting for the heartbeat
            timeout. Defaults to False (PyTorch default behavior).
        nccl_collective_timeout_sec (Optional[int]): Timeout in seconds for individual
            NCCL collective operations (e.g. all-reduce inside loss.backward()). This
            is the timeout passed to ``torch.distributed.init_process_group``. When a
            worker desyncs (e.g. skips a collective after OOM), surviving workers block
            in the collective for this long before the NCCL watchdog fires. This is the
            first phase of failure detection. PyTorch default is 600s (10 min). Set to
            None to use PyTorch default.
    """

    nnodes: Union[int, str]
    nproc_per_node: int
    rdzv_backend: Literal["c10d", "etcd", "etcd-v2"] = "c10d"
    run_policy: Optional[RunPolicy] = None
    monitor_interval: int = 3
    max_restarts: int = 3
    rdzv_configs: Dict[str, Any] = field(default_factory=lambda: {"timeout": 900, "join_timeout": 900})
    nccl_heartbeat_timeout_sec: Optional[int] = 300
    nccl_async_error_handling: bool = False
    nccl_collective_timeout_sec: Optional[int] = None


def launcher_entrypoint(tctx: TaskContext, fn: bytes, kwargs: dict):
    func = cloudpickle.loads(fn)
    flyte.init(
        org=tctx.action.org,
        project=tctx.action.project,
        domain=tctx.action.domain,
        root_dir=tctx.run_base_dir,
    )

    # Override the default NCCL collective timeout before the user calls
    # init_process_group().  We must patch both the constants module AND
    # the distributed_c10d module because `from constants import
    # default_pg_nccl_timeout` creates a separate name binding.
    nccl_timeout = os.environ.get("FLYTE_NCCL_COLLECTIVE_TIMEOUT_SEC")
    if nccl_timeout is not None:
        from datetime import timedelta

        import torch.distributed.constants
        import torch.distributed.distributed_c10d

        td = timedelta(seconds=int(nccl_timeout))
        torch.distributed.constants.default_pg_nccl_timeout = td
        torch.distributed.distributed_c10d.default_pg_nccl_timeout = td

    with internal_ctx().replace_task_context(tctx):
        return func(**kwargs)


_SIGNAL_HINTS = {
    -signal.SIGABRT: (
        "Worker was killed by SIGABRT. This usually means an NCCL collective "
        "timed out — a common symptom of CUDA OOM causing ranks to desync. "
        "Check worker logs for 'CUDA out of memory' or 'collective operation timeout'."
    ),
    -signal.SIGKILL: (
        "Worker was killed by SIGKILL (OOM killer or resource limit). "
        "The process likely exceeded its memory limit. "
        "Try reducing batch size or requesting more memory."
    ),
}


def _format_worker_failure(err: ChildFailedError) -> str:
    """Build a human-readable error message from a ChildFailedError."""
    lines = ["PyTorch worker(s) failed:"]
    for rank, failure in err.failures.items():
        exitcode = failure.exitcode
        hint = _SIGNAL_HINTS.get(exitcode, "")
        lines.append(f"  rank {rank}: exitcode={exitcode} (pid {failure.pid})")
        if hint:
            lines.append(f"    -> {hint}")
        if failure.message and "Signal" not in failure.message:
            lines.append(f"    -> {failure.message}")
    return "\n".join(lines)


def _start_zombie_watchdog(nproc: int, check_interval: float = 10.0) -> threading.Event:
    """Start a daemon thread that detects zombie worker processes and force-exits.

    PyTorch's elastic agent has a known deadlock: when all worker processes die
    from SIGABRT simultaneously (e.g. NCCL abort on collective timeout), the
    agent's _poll() method calls multiprocessing.Event.set() which deadlocks in
    Condition.notify() trying to acquire a shared semaphore that the dead workers
    will never release.  See: torch/distributed/elastic/multiprocessing/api.py
    comment on _worker_finished_event.

    This watchdog periodically discovers child processes via /proc and counts
    how many are zombies.  When at least ``nproc`` children are zombies, the
    workers are all dead and the elastic agent is deadlocked.  We force-exit
    with os._exit() since the deadlocked semaphore cannot be interrupted
    cleanly.

    Note: not all children are workers — Python's multiprocessing spawns a
    ``resource_tracker`` process that stays alive.  We count zombie children
    rather than requiring *all* children to be zombies.
    """
    stop = threading.Event()
    my_pid = os.getpid()

    def _count_zombie_children():
        """Count direct child processes that are zombies."""
        zombie_pids = []
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                pid = int(entry)
                if pid == my_pid:
                    continue
                try:
                    with open(f"/proc/{pid}/status") as f:
                        ppid = None
                        is_zombie = False
                        for line in f:
                            if line.startswith("PPid:"):
                                ppid = int(line.split()[1])
                            elif line.startswith("State:"):
                                is_zombie = "Z" in line
                            if ppid is not None and is_zombie:
                                break
                        if ppid == my_pid and is_zombie:
                            zombie_pids.append(pid)
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    pass
        except OSError:
            pass
        return zombie_pids

    def _run():
        while not stop.wait(check_interval):
            zombie_pids = _count_zombie_children()
            # Once we have at least nproc zombie children, all workers are dead.
            # Other children (e.g. multiprocessing.resource_tracker) may still
            # be alive — that's expected and shouldn't prevent detection.
            if len(zombie_pids) >= nproc:
                logger.error(
                    "Zombie watchdog: %d worker processes are zombies (PIDs %s). "
                    "This indicates a PyTorch elastic agent deadlock in "
                    "multiprocessing.Event.set(). Force-exiting.",
                    len(zombie_pids),
                    zombie_pids,
                )
                os._exit(1)

    t = threading.Thread(target=_run, daemon=True, name="zombie-watchdog")
    t.start()
    return stop


@dataclass(kw_only=True)
class TorchFunctionTask(AsyncFunctionTaskTemplate):
    """
    Plugin to transform local python code for execution as a PyTorch job.
    """

    task_type: str = "pytorch"
    task_type_version: int = 1
    plugin_config: Elastic
    debuggable: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.task_type = "python-task" if self.plugin_config.nnodes == 1 else "pytorch"
        self.min_nodes, self.max_nodes = run.parse_min_max_nnodes(str(self.plugin_config.nnodes))

    async def pre(self, *args: P.args, **kwargs: P.kwargs) -> Dict[str, Any]:
        # If OMP_NUM_THREADS is not set, set it to 1 to avoid overloading the system.
        # Doing so to copy the default behavior of torchrun.
        # See https://github.com/pytorch/pytorch/blob/eea4ece256d74c6f25c1f4eab37b3f2f4aeefd4d/torch/distributed/run.py#L791
        if "OMP_NUM_THREADS" not in os.environ and self.plugin_config.nproc_per_node > 1:
            omp_num_threads = 1
            logger.warning(
                "\n*****************************************\n"
                "Setting OMP_NUM_THREADS environment variable for each process to be "
                "%s in default, to avoid your system being overloaded, "
                "please further tune the variable for optimal performance in "
                "your application as needed. \n"
                "*****************************************",
                omp_num_threads,
            )
            os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

        # Set NCCL heartbeat timeout so surviving workers detect a dead peer
        # (e.g. CUDA OOM crash) faster than the PyTorch default of 1800s.
        if (
            self.plugin_config.nccl_heartbeat_timeout_sec is not None
            and "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC" not in os.environ
        ):
            os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = str(self.plugin_config.nccl_heartbeat_timeout_sec)

        # When enabled, NCCL aborts stuck collectives asynchronously instead of
        # blocking.  The worker process crash-exits, letting the elastic agent
        # detect the failure within monitor_interval (~3s) rather than waiting
        # for the full heartbeat timeout.
        if self.plugin_config.nccl_async_error_handling and "TORCH_NCCL_ASYNC_ERROR_HANDLING" not in os.environ:
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

        # Propagate the collective timeout to worker subprocesses via env var.
        # The launcher_entrypoint reads this and overrides the PyTorch default
        # before user code calls init_process_group().
        if (
            self.plugin_config.nccl_collective_timeout_sec is not None
            and "FLYTE_NCCL_COLLECTIVE_TIMEOUT_SEC" not in os.environ
        ):
            os.environ["FLYTE_NCCL_COLLECTIVE_TIMEOUT_SEC"] = str(self.plugin_config.nccl_collective_timeout_sec)

        return {}

    async def execute(self, *args: P.args, **kwargs: P.kwargs) -> R:
        ctx = internal_ctx()
        tctx = ctx.data.task_context

        if tctx.mode == "local":
            return self.func(*args, **kwargs)

        ctx_data = await self.pre(*args, **kwargs)
        tctx = tctx.replace(data=ctx_data)

        with ctx.replace_task_context(tctx):
            config = LaunchConfig(
                run_id=flyte.ctx().action.run_name,
                min_nodes=self.min_nodes,
                max_nodes=self.max_nodes,
                nproc_per_node=self.plugin_config.nproc_per_node,
                rdzv_backend=self.plugin_config.rdzv_backend,
                rdzv_configs=self.plugin_config.rdzv_configs,
                rdzv_endpoint=os.environ.get("PET_RDZV_ENDPOINT", "localhost:0"),
                max_restarts=self.plugin_config.max_restarts,
                monitor_interval=self.plugin_config.monitor_interval,
            )

            # elastic_launch must run on the main thread so it can register
            # signal handlers (SIGTERM/SIGINT) for cleaning up worker
            # subprocesses.  Running it in a thread pool (run_sync_with_loop)
            # would cause the "Failed to register signal handlers" warning
            # and leave orphaned workers on exit.
            #
            # A zombie watchdog runs in a daemon thread to detect a known
            # PyTorch deadlock: when all workers die from SIGABRT (NCCL abort),
            # the elastic agent deadlocks in multiprocessing.Event.set() →
            # Condition.notify() trying to acquire a shared semaphore that dead
            # workers will never release.  The watchdog force-exits when it
            # detects all worker children are zombies.
            watchdog_stop = _start_zombie_watchdog(nproc=self.plugin_config.nproc_per_node)
            try:
                out = elastic_launch(config=config, entrypoint=launcher_entrypoint)(
                    tctx,
                    cloudpickle.dumps(self.func),
                    kwargs,
                )
            except ChildFailedError as e:
                raise RuntimeError(_format_worker_failure(e)) from e
            finally:
                watchdog_stop.set()

            # `out` is a dictionary of rank (not local rank) -> result
            # Rank 0 returns the result of the task function
            result = out[0] if 0 in out else None
            await self.post(result)

        return result

    def custom_config(self, sctx: SerializationContext) -> Optional[Dict[str, Any]]:
        """
        Converts the ElasticConfig to a DistributedPyTorchTrainingTask
        """
        elastic_config = ElasticConfig(
            rdzv_backend=self.plugin_config.rdzv_backend,
            min_replicas=self.min_nodes,
            max_replicas=self.max_nodes,
            nproc_per_node=self.plugin_config.nproc_per_node,
            max_restarts=self.plugin_config.max_restarts,
        )

        policy = None
        if self.plugin_config.run_policy:
            policy = common_pb2.RunPolicy(
                clean_pod_policy=(
                    # https://github.com/flyteorg/flyte/blob/4caa5639ee6453d86c823181083423549f08f702/flyteidl/protos/flyteidl/plugins/kubeflow/common.proto#L9-L13
                    common_pb2.CleanPodPolicy.Value(
                        "CLEANPOD_POLICY_" + self.plugin_config.run_policy.clean_pod_policy.upper()
                    )
                    if self.plugin_config.run_policy.clean_pod_policy
                    else None
                ),
                ttl_seconds_after_finished=self.plugin_config.run_policy.ttl_seconds_after_finished,
                active_deadline_seconds=self.plugin_config.run_policy.active_deadline_seconds,
                backoff_limit=self.plugin_config.run_policy.backoff_limit,
            )

        torch_job = DistributedPyTorchTrainingTask(
            worker_replicas=DistributedPyTorchTrainingReplicaSpec(
                replicas=self.max_nodes,
            ),
            run_policy=policy,
            elastic_config=elastic_config,
        )

        return MessageToDict(torch_job)


TaskPluginRegistry.register(config_type=Elastic, plugin=TorchFunctionTask)
