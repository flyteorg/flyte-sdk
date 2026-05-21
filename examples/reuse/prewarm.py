"""Pre-warm a reusable env's worker pool before the first heavy task fires.

Demonstrates the cold-start hide via `env.prewarm()`. Two driver tasks
exercise the same reusable env:

  * ``with_prewarm``    — schedules `heavy_env.prewarm()` with
    ``asyncio.create_task``, does N seconds of unrelated async work, then
    calls ``heavy_task``. The pool warms in parallel with the setup work
    so the heavy task lands on HEALTHY workers.
  * ``without_prewarm`` — same shape, no prewarm. ``heavy_task`` waits in
    ``WaitingForResources`` until the first pod is HEALTHY.

To make the cold-start cost visible, ``heavy_env`` attaches a pod template
whose init container just sleeps. K8s blocks the primary container until
init containers finish, so every first-time-on-this-node pod pays the
``INIT_SLEEP_SECONDS`` cost. Once warm, the pool reuses those pods for
free.

Compare the two runs in the UI: the gap between ``submitted`` and
``started`` on ``heavy_task`` is near-zero with prewarm and
``~INIT_SLEEP_SECONDS`` without.
"""

import asyncio

from kubernetes.client import V1Container, V1PodSpec

import flyte


# Seconds the init container will sleep. Stand-in for image pull / model
# load / any first-pod startup cost. Tune to ≈ the driver's ``setup_seconds``
# so the benefit of prewarm is clearly visible but not exaggerated.
INIT_SLEEP_SECONDS = 60


# If you're working against an unreleased SDK build, swap this image
# definition for one that bakes the local wheel:
#
#   image = (
#       flyte.Image.from_debian_base(install_flyte=False)
#       .with_pip_packages("unionai-reuse", "kubernetes")
#       .with_local_v2()
#   )
#
# requires ``python -m build --wheel`` in the flyte-sdk repo so a wheel
# exists in ``dist/``.
image = flyte.Image.from_debian_base().with_pip_packages("unionai-reuse", "kubernetes")


# Init container blocks the primary container until it finishes —
# simulates a slow first-pod startup so prewarm has something meaningful
# to hide.
slow_startup_pod = flyte.PodTemplate(
    primary_container_name="primary",
    pod_spec=V1PodSpec(
        containers=[V1Container(name="primary")],
        init_containers=[
            V1Container(
                name="slow-startup-sim",
                image="busybox:latest",
                command=["sh", "-c", f"echo 'simulating slow startup'; sleep {INIT_SLEEP_SECONDS}"],
            ),
        ],
    ),
)


heavy_env = flyte.TaskEnvironment(
    name="prewarm_demo_heavy",
    resources=flyte.Resources(cpu=1, memory="500Mi"),
    image=image,
    pod_template=slow_startup_pod,
    reusable=flyte.ReusePolicy(
        replicas=(2, 2),
        # Sized to cover the driver's pre-heavy work plus a margin. The
        # same idle_ttl also applies after the heavy task completes, so a
        # longer value also delays scale-down.
        idle_ttl=600,
        scaledown_ttl=60,
    ),
)


# driver_env is defined separately (not via clone_with) so it does NOT
# inherit heavy_env's slow-startup pod_template — we only want the heavy
# pods to pay the init-container cost.
driver_env = flyte.TaskEnvironment(
    name="prewarm_demo_driver",
    resources=flyte.Resources(cpu=1, memory="500Mi"),
    image=image,
    depends_on=[heavy_env],
)


@heavy_env.task
async def heavy_task(x: int) -> int:
    # Cheap once a worker is alive; the interesting cost is reaching this
    # point from a cold pool (pod schedule + init container + start).
    print(f"heavy_task running with x={x}")
    return x * 2


@driver_env.task
async def with_prewarm(setup_seconds: int = 90) -> int:
    """Fire-and-forget prewarm via asyncio.create_task, then do unrelated work.

    ``await heavy_env.prewarm()`` would block until the pool is HEALTHY,
    defeating the parallelism. The Pythonic fire-and-forget pattern is
    ``asyncio.create_task`` — schedule it on the event loop, let it run
    while we await other things.
    """
    print("scheduling prewarm() — pool warms in background")
    asyncio.create_task(heavy_env.prewarm())

    print(f"simulating {setup_seconds}s of pre-heavy work")
    await asyncio.sleep(setup_seconds)

    print("now calling heavy_task — pool should already be HEALTHY")
    return await heavy_task(21)


@driver_env.task
async def without_prewarm(setup_seconds: int = 90) -> int:
    """Baseline: same shape, no prewarm. heavy_task hits a cold pool."""
    print(f"simulating {setup_seconds}s of pre-heavy work (no prewarm)")
    await asyncio.sleep(setup_seconds)

    print("now calling heavy_task — pool is cold; pays init-container cost")
    return await heavy_task(21)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(with_prewarm, setup_seconds=90)
    # run = flyte.run(without_prewarm, setup_seconds=90)
    print("run url:", run.url)
    run.wait()
