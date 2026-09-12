"""
Multi-process JAX on a MultiNodeTaskEnvironment(runtime=JaxRun()).

Runs one JAX process per pod across ``replicas`` pods, bootstrapped by the dedicated ``clustered``
runtime entrypoint (``--runtime=jax``): it exports the process topology from the JobSet env vars
and execs ``a0`` directly — no torchrun and no torch in the image. Each process calls
``flyte.clustered.jax_initialize()`` (``jax.distributed.initialize`` with the coordinator on pod 0),
then all-gathers its rank across the whole gang, so a correct result proves real cross-process
communication. CPU-only, so it runs on any cluster with the clustered plugin.

This exercises the full path:
    MultiNodeTaskEnvironment(runtime=JaxRun())  ->  task_serde (args -> `clustered --runtime=jax`)
      ->  JobSet (N pods)  ->  `clustered` entrypoint: DNS wait + exec a0  ->  1 process per pod
      ->  jax.distributed coordinator rendezvous  ->  process_allgather  ->  rank-0 uploads outputs.

Run (registers + runs on the configured cluster):
    uv run python examples/clustered/jax_allgather.py
"""

from __future__ import annotations

import flyte
from flyte._image import DIST_FOLDER, PythonWheels
from flyte.clustered import ClusterFailurePolicy, JaxRun, MultiNodeTaskEnvironment, jax_initialize

# Image carries the LOCAL flyte build (so the container has the `clustered` runtime entrypoint
# with the jax launcher), plus CPU jax for the workload.
image = (
    flyte.Image.from_debian_base(name="jax_allgather1")
    .clone(addl_layer=PythonWheels(wheel_dir=DIST_FOLDER, package_name="flyte"))
    .with_pip_packages("jax")
)

REPLICAS = 2  # pods == JAX processes (JaxRun runs one process per pod)

env = MultiNodeTaskEnvironment(
    name="jax_env",
    image=image,
    resources=flyte.Resources(cpu=(1, 2), memory=("1Gi", "2Gi")),
    replicas=REPLICAS,
    nproc_per_node=1,
    runtime=JaxRun(),
    # Cross-process CPU collectives need an implementation. Current JAX defaults to gloo but older
    # releases defaulted to none, so pin it (JAX reads its config flags from JAX_<FLAG> env vars).
    env_vars={"JAX_CPU_COLLECTIVES_IMPLEMENTATION": "gloo"},
    failure_policy=ClusterFailurePolicy(max_restarts=1),
)


@env.task
async def allgather_ranks() -> int:
    """All-gather every process's rank across the gang and return their sum (uploaded by rank 0)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import multihost_utils

    ctx = flyte.ctx()
    jax_initialize()

    rank, world_size = jax.process_index(), jax.process_count()
    print(
        f"[proc {rank}/{world_size}] ctx.rank={ctx.rank} ctx.world_size={ctx.world_size} "
        f"ctx.node_rank={ctx.node_rank} master_addr={ctx.master_addr} restart_attempt={ctx.restart_attempt} "
        f"local_devices={jax.local_device_count()} global_devices={jax.device_count()}",
        flush=True,
    )
    assert world_size == ctx.world_size, (world_size, ctx.world_size)
    assert rank == ctx.rank, (rank, ctx.rank)
    assert jax.device_count() == world_size * jax.local_device_count()

    multihost_utils.sync_global_devices("start")
    gathered = multihost_utils.process_allgather(jnp.array([rank], dtype=jnp.int32))
    ranks = [int(r) for r in gathered.reshape(-1)]
    print(f"[proc {rank}] gathered ranks: {ranks}", flush=True)
    assert ranks == list(range(world_size)), ranks

    multihost_utils.sync_global_devices("done")
    return sum(ranks)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(allgather_ranks)
    print("Run URL:", run.url)
    run.wait()
    print("Final phase:", run.phase)
