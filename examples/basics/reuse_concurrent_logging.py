"""
Demo: reusable container + concurrent tasks, so log lines from different
actions are interleaved in the same worker process. Each line should be
prefixed with its own [run][action] thanks to the LogRecord factory in
flyte._logging.

Workers exercise three logger configs to confirm context stamping works
regardless of how the user obtains a logger:

  1. flyte.logger                           — the canonical user-facing logger
  2. logging.getLogger("flyte.user.myapp")  — child of flyte.user, inherits handler
  3. logging.getLogger("myapp")             — fully independent logger w/ its own handler
"""

import asyncio
import logging
import sys

import flyte

# One worker process, many concurrent invocations on it. With replicas=1 and
# concurrency=8, all worker tasks below land in the same Python process, so
# their log output is genuinely interleaved on a single stderr.
env = flyte.TaskEnvironment(
    name="reuse_concurrent_logging",
    resources=flyte.Resources(cpu="1", memory="500Mi"),
    reusable=flyte.ReusePolicy(
        replicas=1,
        concurrency=8,
        idle_ttl=60,
        scaledown_ttl=60,
    ),
    image=flyte.Image.from_debian_base().with_pip_packages("unionai-reuse"),
)

# Variant 2: child of flyte.user. No handler/level setup needed — propagation
# carries records up to flyte.user's handler, so [run][action] prefixing and
# the user log level both apply automatically.
inherited_logger = logging.getLogger("flyte.user.myapp")

# Variant 3: independent logger with its own StreamHandler. The flyte
# LogRecordFactory still stamps run_name/action_name on every record, so the
# formatter below can reference them via %(run_name)s / %(action_name)s even
# though this logger lives outside the flyte.* namespace.
independent_logger = logging.getLogger("myapp")
if not independent_logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[%(run_name)s][%(action_name)s] myapp: %(message)s"))
    independent_logger.addHandler(_h)
    independent_logger.setLevel(logging.INFO)
    independent_logger.propagate = False


@env.task
async def worker_flyte_logger(label: str, ticks: int = 3) -> str:
    """Variant 1: the canonical flyte.logger."""
    flyte.logger.info("starting label=%s", label)
    for i in range(ticks):
        await asyncio.sleep(1)
        flyte.logger.info("label=%s tick=%d/%d", label, i + 1, ticks)
    flyte.logger.info("done label=%s", label)
    return label


@env.task
async def worker_inherited_logger(label: str, ticks: int = 3) -> str:
    """Variant 2: child of flyte.user — formatting is inherited."""
    inherited_logger.info("starting label=%s", label)
    for i in range(ticks):
        await asyncio.sleep(1)
        inherited_logger.info("label=%s tick=%d/%d", label, i + 1, ticks)
    inherited_logger.info("done label=%s", label)
    return label


@env.task
async def worker_independent_logger(label: str, ticks: int = 3) -> str:
    """Variant 3: a fully independent stdlib logger — record factory still stamps context."""
    independent_logger.info("starting label=%s", label)
    for i in range(ticks):
        await asyncio.sleep(1)
        independent_logger.info("label=%s tick=%d/%d", label, i + 1, ticks)
    independent_logger.info("done label=%s", label)
    return label


# The fan-out parent itself doesn't need to share the reuse pool — clone_with
# turns reuse off for it and depends on the worker env for image + registration.
@env.clone_with(name="reuse_concurrent_main", reusable=None, depends_on=[env]).task
async def main(n: int = 2) -> list[str]:
    """
    Dispatches a mix of all three worker variants so logs from each logger
    config interleave on the reused container's stderr.
    """
    flyte.logger.info("dispatching %d of each worker variant", n)
    coros: list = []
    for i in range(n):
        coros.append(worker_flyte_logger(label=f"flyte-{i}"))
        coros.append(worker_inherited_logger(label=f"inherited-{i}"))
        coros.append(worker_independent_logger(label=f"independent-{i}"))
    results = await asyncio.gather(*coros)
    flyte.logger.info("all workers finished: %s", results)
    return results


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, n=2)
    print(run.url)
    run.wait()
