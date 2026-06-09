"""
Entrypoint for clustered (JobSet-based) distributed training pods.

Kept separate from the hot `flyte._bin.runtime` module. Exposes the `clustered` console script
(see pyproject `[project.scripts]`) which the Go `clustered` plugin wires into a Kubernetes JobSet.

Caution: like `runtime`, startup time matters here — keep top-level imports minimal and import
heavier modules inside functions.
"""

import os
import sys
from typing import List

import click

from flyte._bin.runtime import _action_options, _run_action

# --- clustered (JobSet) launcher constants ---
_DNS_TIMEOUT_SEC = 300
_DNS_RETRY_INTERVAL_SEC = 2
_CLUSTERED_REQUIRED_ENV_VARS = (
    "JOBSET_NAME",
    "POD_NAMESPACE",
    "JOB_COMPLETION_INDEX",
    "JOBSET_RESTART_ATTEMPT",
    "NNODES",
    "NPROC_PER_NODE",
    "RDZV_BACKEND",
)


@click.command("clustered")
@_action_options
@click.pass_context
def clustered_main(ctx: click.Context, **params):
    """Entrypoint for clustered (JobSet-based) distributed training pods.

    Two phases in one command:
      * Launcher (PID 1, no torchrun env): derive the torchrun rendezvous from JobSet env vars and
        exec `torchrun ... -- clustered <same args>`.
      * Worker (re-entered by torchrun, TORCHELASTIC_RUN_ID set): run the task. A clustered task
        never enqueues subtasks, so it runs with no controller; outputs/errors upload via storage.
    """
    if "TORCHELASTIC_RUN_ID" in os.environ:
        _run_action(ctx, controller_enabled=False, **params)
        return
    _exec_torchrun_launcher(worker_argv=["clustered", *sys.argv[1:]])


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

    for var in _CLUSTERED_REQUIRED_ENV_VARS:
        if not os.environ.get(var):
            logger.error(f"required env var {var!r} is not set")
            sys.exit(1)

    jobset_name = os.environ["JOBSET_NAME"]
    namespace = os.environ["POD_NAMESPACE"]
    node_rank = os.environ["JOB_COMPLETION_INDEX"]
    restart_attempt = os.environ["JOBSET_RESTART_ATTEMPT"]
    nnodes = os.environ["NNODES"]
    nproc_per_node = os.environ["NPROC_PER_NODE"]
    rdzv_backend = os.environ["RDZV_BACKEND"]
    master_port = os.environ.get("MASTER_PORT", "29500")

    if shutil.which("torchrun") is None:
        logger.error("please install torchrun")
        sys.exit(1)

    master_addr = _master_addr(jobset_name, namespace)
    rdzv_id = f"{jobset_name}-{restart_attempt}"

    _wait_for_dns(master_addr)

    # Export derived vars so torchrun child processes (and the task / TaskContext) inherit them.
    # torchrun does not propagate NODE_RANK to workers, and MASTER_ADDR is only computed here.
    os.environ["NODE_RANK"] = node_rank
    os.environ["MASTER_ADDR"] = master_addr

    torchrun_cmd = [
        "torchrun",
        f"--nnodes={nnodes}",
        f"--nproc-per-node={nproc_per_node}",
        f"--node-rank={node_rank}",
        f"--rdzv-backend={rdzv_backend}",
        f"--rdzv-id={rdzv_id}",
        f"--rdzv-endpoint={master_addr}:{master_port}",
        # worker_argv[0] is the `clustered` console-script executable (flyte._bin.clustered:clustered_main),
        # not a .py file. Without --no-python torchrun would run `python clustered` and fail. --no-python
        # makes torchrun exec the command directly.
        "--no-python",
        "--",
        *worker_argv,
    ]

    logger.info(f"exec: {' '.join(torchrun_cmd)}")
    os.execvp("torchrun", torchrun_cmd)


if __name__ == "__main__":
    clustered_main()
