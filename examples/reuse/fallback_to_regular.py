"""
Example: run a task on a reusable (actor) container, and on exception fall back
to a fresh, regular (non-reusable) container with bigger resources.

Why you might want this:
- Reusable containers are fast (no cold start) but share memory/state across
  invocations on the same replica. A pathological input that OOMs or corrupts
  process state can make subsequent calls flaky.
- Falling back to a regular container gives you a clean process *and* lets you
  bump resources, since `override(reusable="off", resources=...)` is allowed
  but `override(resources=...)` on a reusable task is not.
"""

import flyte
import flyte.errors

actor_image = flyte.Image.from_debian_base().with_pip_packages("unionai-reuse")

env = flyte.TaskEnvironment(
    name="reuse_with_fallback",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
    image=actor_image,
    reusable=flyte.ReusePolicy(replicas=2, idle_ttl=60),
)


@env.task
async def crunch(x: int) -> int:
    # Simulate workloads of varying size. Small inputs fit comfortably in the
    # reusable container's 250Mi; large inputs blow past it.
    # Allocate ~x * 50MiB and touch every page so the kernel actually backs it.
    chunk = 50 * 1024 * 1024
    buffers = []
    for _ in range(x):
        b = bytearray(chunk)
        # Touch pages to force real allocation (avoid lazy/overcommit).
        for i in range(0, chunk, 4096):
            b[i] = 1
        buffers.append(b)
    return sum(len(b) for b in buffers)


@env.task
async def driver(x: int) -> int:
    try:
        # Fast path: reusable container, no cold start.
        return await crunch(x)
    except (flyte.errors.OOMError, flyte.errors.RuntimeUserError) as e:
        print(f"Reusable run failed ({type(e).__name__}: {e.code}); retrying on a regular container with more memory")
        # Fallback: disable reuse and request a fresh, larger container.
        # `reusable="off"` is required before you can override resources on a
        # task whose env is reusable.
        return await crunch.override(
            reusable="off",
            resources=flyte.Resources(cpu=1, memory="2Gi"),
        )(x)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(driver, x=20)
    print(run.url)
    run.wait()
