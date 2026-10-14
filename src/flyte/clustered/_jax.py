"""JAX helpers for `ClusteredTaskEnvironment(runtime=JaxRun())`."""

from __future__ import annotations

from typing import Any, Dict


def jax_initialize(**overrides: Any) -> None:
    """Initialize `jax.distributed` for this clustered task's process topology.

    Wraps `jax.distributed.initialize` with the coordinator address, process count and process id
    that the `clustered` launcher exported for a `JaxRun` environment, and disables JAX's cluster
    auto-detection: its Kubernetes detector otherwise activates in every pod and either fails to
    import the `kubernetes` client or queries the API without RBAC. Any keyword argument is forwarded
    to `jax.distributed.initialize` and wins over the defaults, e.g. `local_device_ids=[0]`.

    Safe to call more than once: subsequent calls are no-ops once JAX reports it is initialized.

    Raises:
        RuntimeError: when called outside a `JaxRun` clustered task (no process topology in the
            environment).
    """
    import jax

    import flyte

    is_initialized = getattr(jax.distributed, "is_initialized", None)
    if is_initialized is not None and is_initialized():
        return

    # flyte.ctx() returns a null context outside a task; its topology fields read the environment
    # either way, so this works in any process the `clustered` launcher started.
    ctx = flyte.ctx()
    rank, world_size = ctx.rank, ctx.world_size
    master_addr, master_port = ctx.master_addr, ctx.master_port
    if rank is None or world_size is None or master_addr is None or master_port is None:
        raise RuntimeError(
            "jax_initialize() needs the process topology exported by the `clustered` launcher; "
            "run this task on a ClusteredTaskEnvironment(runtime=JaxRun())"
        )

    params: Dict[str, Any] = {
        "coordinator_address": f"{master_addr}:{master_port}",
        "num_processes": world_size,
        "process_id": rank,
        "cluster_detection_method": "deactivate",
    }
    params.update(overrides)
    jax.distributed.initialize(**params)
