"""
Multi-cluster queue distribution test — ENG26-923 (prereq for ENG26-922).

Fans out N sustained actions onto queue dogfood-mc-test, which spansdogfood-1 + dogfood-3 (both in the `efault`cluster pool). Verifies:  1. Distribution — actions land on BOTH clusters (leasor routes at the
     action level, picking a worker from the queue's cluster list).
  2. Correctness — cross-cluster data passing works: combine() reads every
     work() output, which for actions that ran on the other cluster means a
     read through the shared metadata bucket (union-cloud-dogfood-1-dogfood).

Control plane: dogfood org, STAGING -> dns:///dogfood.cloud-staging.union.ai
IMPORTANT: this must run against the dogfood staging config, NOT the default
~/.union/config.yaml (which points at a different tenant). CONFIG below pins it.

Run (programmatic — targets the queue via with_runcontext):
    python mc_queue_test.py

Or via CLI (queue comes from the TaskEnvironment default):
    flyte --config ~/.uctl/config-dogfood.yaml run \
        --project <PROJECT> --domain <DOMAIN> mc_queue_test.py main --n 100
"""

import asyncio
import os

import flyte

QUEUE = "dogfood-mc-test"
CONFIG = os.environ.get("FLYTE_CONFIG", os.path.expanduser("~/.flyte/dogfood.staging.yaml"))
PROJECT = os.environ.get("FLYTE_PROJECT", "flytesnacks")
DOMAIN = os.environ.get("FLYTE_DOMAIN", "development")
N = int(os.environ.get("N", "100"))

# A non-default dependency (numpy) forces a real remote image build instead of
# reusing the prebuilt ghcr base. The build routes by pool, so it can land on
# either cluster's buildkit and is pushed to the shared user ECR (union/dogfood)
# -> exercises the create_user_repository/shared-ECR fix end to end.
image = flyte.Image.from_debian_base().with_pip_packages("numpy")

# queue on the env => every task (and a CLI-launched run) targets dogfood-mc-test
env = flyte.TaskEnvironment(name="mc_queue_test", queue=QUEUE, image=image)


@env.task
async def work(i: int) -> int:
    import numpy as np

    # 90s keeps both clusters saturated at once so the split reflects real
    # distribution, not a race won by whichever worker polls first.
    await asyncio.sleep(90)
    return int(np.multiply(i, 2))


@env.task
async def combine(xs: list[int]) -> int:
    import numpy as np

    # Reads every work() output. For actions that ran on the OTHER cluster this
    # is a cross-cluster read via the shared metadata bucket -> exercises the
    # core reason both clusters must share a bucket.
    return int(np.sum(xs))


@env.task
async def main(n: int = 100) -> int:
    results = await asyncio.gather(*[work(i) for i in range(n)])
    return await combine(results)


if __name__ == "__main__":
    # image_builder="remote" => build on the cluster's buildkit and push to the
    # shared union/dogfood ECR (no local Docker daemon needed). Without this the
    # SDK defaults to the local docker builder + ghcr, which is not the path we
    # want to exercise here.
    flyte.init_from_config(CONFIG, project=PROJECT, domain=DOMAIN, image_builder="remote")
    run = flyte.with_runcontext(queue=QUEUE).run(main, n=N)
    print(f"run:  {run.name}")
    print(f"url:  {run.url}")