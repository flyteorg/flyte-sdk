"""
Entrypoint for clustered (JobSet-based) distributed training pods.

Runs as PID 1 via: python -m flyte.clustered._entrypoint <a0 args...>

Responsibilities:
  1. Validate required env vars
  2. Derive MASTER_ADDR from JOBSET_NAME + POD_NAMESPACE
  3. Busy-wait for DNS resolution (5-min timeout)
  4. Map JOB_COMPLETION_INDEX -> NODE_RANK
  5. Derive RDZV_ID = JOBSET_NAME-JOBSET_RESTART_ATTEMPT
  6. exec torchrun, passing sys.argv[1:] (the a0 invocation) as the worker command
"""

import os
import shutil
import socket
import sys
import time

from flyte._logging import logger

_DNS_TIMEOUT_SEC = 300
_DNS_RETRY_INTERVAL_SEC = 2

_REQUIRED_ENV_VARS = [
    "JOBSET_NAME",
    "POD_NAMESPACE",
    "JOB_COMPLETION_INDEX",
    "JOBSET_RESTART_ATTEMPT",
    "NNODES",
    "NPROC_PER_NODE",
    "RDZV_BACKEND",
]


def _require(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        logger.error(f"required env var {var!r} is not set")
        sys.exit(1)
    return val


def _master_addr(jobset_name: str, namespace: str) -> str:
    return f"{jobset_name}-workers-0-0.{jobset_name}.{namespace}.svc.cluster.local"


def _wait_for_dns(hostname: str, timeout: float = _DNS_TIMEOUT_SEC, interval: float = _DNS_RETRY_INTERVAL_SEC) -> None:
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


def main() -> None:
    for var in _REQUIRED_ENV_VARS:
        _require(var)

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

    # Export derived vars so torchrun child processes (and a0 / TaskContext) inherit them.
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
        # sys.argv[1:] starts with the `a0` console-script executable (flyte._bin.runtime:main),
        # not a .py file. Without --no-python torchrun would run `python a0` and fail with
        # "can't open file 'a0'". --no-python makes torchrun exec the command directly.
        "--no-python",
        "--",
        *sys.argv[1:],
    ]

    logger.info(f"exec: {' '.join(torchrun_cmd)}")
    os.execvp("torchrun", torchrun_cmd)


if __name__ == "__main__":
    main()
