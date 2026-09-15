"""
Entrypoint for clustered (JobSet-based) distributed training pods.

Kept separate from the hot `flyte._bin.runtime` module. Exposes the `clustered` console script
(see pyproject `[project.scripts]`) which the Go `clustered` plugin wires into a Kubernetes JobSet.

The SDK picks the runtime at serialization time through `--runtime` (see
`flyte.clustered._task.ClusteredTaskTemplate.container_args`). The JobSet env vars the plugin injects
(NNODES, NPROC_PER_NODE, MASTER_PORT, JOBSET_*, POD_NAMESPACE) are shared by every runtime; only
RDZV_BACKEND is torchrun-specific.

Caution: like `runtime`, startup time matters here — keep top-level imports minimal and import
heavier modules inside functions.
"""

import os
import sys
from typing import Dict, List, Sequence

import click

from flyte._bin.runtime import _action_options
from flyte.models import CLUSTERED_WORKER_ENV

# --- clustered (JobSet) launcher constants ---
_DNS_TIMEOUT_SEC = 300
_DNS_RETRY_INTERVAL_SEC = 2
_RUNTIME_OPTION = "--runtime"
_RUNTIMES = ("torchrun", "jax")
# Injected by the Go `clustered` plugin for every runtime (JOB_COMPLETION_INDEX comes from kubelet).
_CLUSTERED_REQUIRED_ENV_VARS = (
    "JOBSET_NAME",
    "POD_NAMESPACE",
    "JOB_COMPLETION_INDEX",
    "JOBSET_RESTART_ATTEMPT",
    "NNODES",
    "NPROC_PER_NODE",
)
_TORCHRUN_REQUIRED_ENV_VARS = (*_CLUSTERED_REQUIRED_ENV_VARS, "RDZV_BACKEND")


@click.command("clustered")
@click.option(
    _RUNTIME_OPTION,
    type=click.Choice(list(_RUNTIMES)),
    default="torchrun",
    help="Runtime to exec in this pod; set by the SDK from ClusteredTaskEnvironment.runtime.",
)
@_action_options
def main(runtime: str, **params):
    """Launcher for clustered (JobSet-based) distributed training pods.

    Runs as the container PID 1. Derives the process topology from JobSet env vars and execs the
    selected runtime with `a0 <same args>` as the worker command, so each worker is the standard
    `a0` runtime entrypoint. Every worker carries `FLYTE_CLUSTERED_WORKER=1` (torchrun also sets
    `TORCHELASTIC_RUN_ID`) and runs with no controller — a clustered task never enqueues subtasks;
    outputs/errors upload via storage.

    The action options are declared (via `_action_options`) only to fail fast on missing required
    args; they are forwarded verbatim to `a0` through `sys.argv` (minus `--runtime`) rather than
    read from `params`.
    """
    worker_argv = ["a0", *_strip_option(sys.argv[1:], _RUNTIME_OPTION)]
    if runtime == "jax":
        _exec_jax_launcher(worker_argv)
    else:
        _exec_torchrun_launcher(worker_argv)


def _strip_option(argv: Sequence[str], name: str) -> List[str]:
    """Drop every `name=value` / `name value` occurrence of a click option from `argv`."""
    stripped: List[str] = []
    skip_value = False
    for token in argv:
        if skip_value:
            skip_value = False
            continue
        if token == name:
            skip_value = True
            continue
        if token.startswith(f"{name}="):
            continue
        stripped.append(token)
    return stripped


def _require_env(names: Sequence[str]) -> Dict[str, str]:
    """Return the values of `names` from the environment, exiting 1 if any is missing or empty."""
    from flyte._logging import logger

    values: Dict[str, str] = {}
    for var in names:
        value = os.environ.get(var)
        if not value:
            logger.error(f"required env var {var!r} is not set")
            sys.exit(1)
        values[var] = value
    return values


def _master_addr(jobset_name: str, namespace: str) -> str:
    return f"{jobset_name}-workers-0-0.{jobset_name}.{namespace}.svc.cluster.local"


def _wait_for_dns(hostname: str, timeout: float = _DNS_TIMEOUT_SEC, interval: float = _DNS_RETRY_INTERVAL_SEC) -> None:
    import socket
    import time

    from flyte._logging import logger

    deadline = time.monotonic() + timeout
    while True:
        try:
            socket.getaddrinfo(hostname, None)
            logger.info(f"DNS resolved: {hostname}")
            return
        except socket.gaierror:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.error(
                    f"DNS for {hostname!r} did not resolve within {timeout}s. "
                    "Check that the JobSet headless service is created and pod-0 is running."
                )
                sys.exit(1)
            time.sleep(min(interval, remaining))


def _exec_torchrun_launcher(worker_argv: List[str]) -> None:
    """Derive the torchrun rendezvous from JobSet env vars and exec torchrun with `worker_argv`.

    Runs as the container PID 1; `os.execvp` replaces this process with torchrun, which re-spawns
    `worker_argv` per process.
    """
    import shutil

    from flyte._logging import logger

    env = _require_env(_TORCHRUN_REQUIRED_ENV_VARS)
    jobset_name = env["JOBSET_NAME"]
    namespace = env["POD_NAMESPACE"]
    node_rank = env["JOB_COMPLETION_INDEX"]
    restart_attempt = env["JOBSET_RESTART_ATTEMPT"]
    nnodes = env["NNODES"]
    nproc_per_node = env["NPROC_PER_NODE"]
    rdzv_backend = env["RDZV_BACKEND"]
    master_port = os.environ.get("MASTER_PORT", "29500")

    if shutil.which("torchrun") is None:
        logger.error("please install torchrun")
        sys.exit(1)

    master_addr = _master_addr(jobset_name, namespace)
    rdzv_id = f"{jobset_name}-{restart_attempt}"

    _wait_for_dns(master_addr)

    # Export derived vars so torchrun child processes (and the task / TaskContext) inherit them.
    # torchrun does not propagate NODE_RANK to workers, and MASTER_ADDR is only computed here.
    # The worker marker rides along the same way (torchrun copies os.environ into every worker).
    os.environ["NODE_RANK"] = node_rank
    os.environ["MASTER_ADDR"] = master_addr
    os.environ[CLUSTERED_WORKER_ENV] = "1"

    torchrun_cmd = [
        "torchrun",
        f"--nnodes={nnodes}",
        f"--nproc-per-node={nproc_per_node}",
        f"--node-rank={node_rank}",
        f"--rdzv-backend={rdzv_backend}",
        f"--rdzv-id={rdzv_id}",
        f"--rdzv-endpoint={master_addr}:{master_port}",
        # worker_argv[0] is the `a0` console-script executable (flyte._bin.runtime:main), not a .py
        # file. Without --no-python torchrun would run `python a0` and fail. --no-python makes
        # torchrun exec the command directly.
        "--no-python",
        "--",
        *worker_argv,
    ]

    logger.info(f"exec: {' '.join(torchrun_cmd)}")
    os.execvp("torchrun", torchrun_cmd)


def _exec_jax_launcher(worker_argv: List[str]) -> None:
    """Derive the JAX process topology from JobSet env vars and exec `worker_argv` directly.

    JAX has no launcher binary: each pod runs exactly one process (`nproc_per_node == 1`, enforced
    by the SDK and re-checked here) that owns every local accelerator, and the task calls
    `flyte.clustered.jax_initialize()` (a thin wrapper over `jax.distributed.initialize`) with the
    topology exported below. Process 0 (pod index 0) hosts the coordinator on
    `MASTER_ADDR:MASTER_PORT`; the other processes retry connecting until JAX's
    `initialization_timeout`.

    Runs as the container PID 1; `os.execvp` replaces this process with `a0`. `a0` therefore inherits
    PID 1 with CPython's default SIGTERM disposition, so on pod deletion (JobSet restart, abort) the
    kubelet waits out `terminationGracePeriodSeconds` before SIGKILL — the same behaviour as a plain
    `a0` task (torchrun's agent handles SIGTERM itself).
    """
    from flyte._logging import logger

    env = _require_env(_CLUSTERED_REQUIRED_ENV_VARS)
    nproc_per_node = env["NPROC_PER_NODE"]
    if nproc_per_node != "1":
        logger.error(f"the jax runtime runs one process per pod; NPROC_PER_NODE must be 1, got {nproc_per_node!r}")
        sys.exit(1)

    jobset_name = env["JOBSET_NAME"]
    namespace = env["POD_NAMESPACE"]
    node_rank = env["JOB_COMPLETION_INDEX"]
    nnodes = env["NNODES"]
    master_port = os.environ.get("MASTER_PORT", "29500")

    master_addr = _master_addr(jobset_name, namespace)
    _wait_for_dns(master_addr)

    # One process per pod: global rank == node rank and world size == number of pods. Exported under
    # the names torchrun uses so `flyte.ctx()` and the rank-0 output/error gates work unchanged.
    # JAX_COORDINATOR_ADDRESS is what jax.distributed.initialize() reads when coordinator_address is
    # omitted; jax_initialize() passes it explicitly as well.
    os.environ["NODE_RANK"] = node_rank
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = node_rank
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = nnodes
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    os.environ["JAX_COORDINATOR_ADDRESS"] = f"{master_addr}:{master_port}"
    os.environ[CLUSTERED_WORKER_ENV] = "1"

    logger.info(
        f"exec: {' '.join(worker_argv)} (jax process {node_rank}/{nnodes}, coordinator {master_addr}:{master_port})"
    )
    os.execvp(worker_argv[0], worker_argv)


if __name__ == "__main__":
    main()
